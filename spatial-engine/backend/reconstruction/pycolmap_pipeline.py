"""
Video -> metric, gravity-aligned scene using pycolmap (pip-installable COLMAP, BSD).

Outputs in data/scenes/<scene_id>/artifacts/:
    points.ply          aligned, coloured sparse point cloud (renders in the web viewer)
    collision_mesh.glb  voxel-surface mesh + floor patch, Y-up, floor at Y=0
    cameras.json        aligned camera path (used for the default fly-through)
    scene_report.json   registration stats for the SOP / abstract
"""
import json
import logging
from pathlib import Path

import numpy as np
import trimesh

from backend.config import DATA_ROOT
from backend.ingestion.frame_selector import SmartFrameSelector
from backend.reconstruction.gravity_aligner import MetricGravityAligner

logger = logging.getLogger("PyColmapPipeline")


def _mapper_options():
    """Relaxed registration thresholds: handheld walkthroughs have many moderately textured
    frames (white walls, glossy floors) that the defaults reject, which splits the model."""
    import pycolmap

    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 12
    opts.init_num_trials = 50
    opts.mapper.abs_pose_min_num_inliers = 12
    opts.mapper.abs_pose_min_inlier_ratio = 0.15
    opts.mapper.init_min_num_inliers = 60
    return opts


def run_sfm(keyframes_dir: Path, work_dir: Path, overlap: int = 15, exhaustive_max: int = 300):
    import pycolmap

    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    fo = pycolmap.FeatureExtractionOptions()
    fo.sift.max_num_features = 8000
    fo.sift.domain_size_pooling = True
    pycolmap.extract_features(db, keyframes_dir, camera_mode=pycolmap.CameraMode.SINGLE, extraction_options=fo)
    # Exhaustive matching links views across rooms and when the camera turns back; sequential
    # matching only links neighbours, which fragments walkthroughs. Fall back for long captures.
    if len(list(keyframes_dir.glob("*.jpg"))) <= exhaustive_max:
        pycolmap.match_exhaustive(db)
    else:
        pycolmap.match_sequential(
            db, pairing_options=pycolmap.SequentialPairingOptions(overlap=overlap, quadratic_overlap=True)
        )
    recs = pycolmap.incremental_mapping(db, keyframes_dir, work_dir, options=_mapper_options())
    if not recs:
        raise RuntimeError("COLMAP could not build any model - capture has too little overlap.")
    # rank by triangulated structure, not image count: many images can sit on a degenerate model
    best_key = max(recs, key=lambda k: recs[k].num_points3D())
    return recs[best_key], len(recs), best_key


def extract_geometry(rec):
    """Return points (N,3), colours (N,3 uint8), camera centres (M,3), c2w rotations (M,3,3)."""
    pts, cols = [], []
    for p in rec.points3D.values():
        pts.append(p.xyz)
        cols.append(p.color)
    centers, rots = [], []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        c2w = img.cam_from_world().inverse().matrix()  # 3x4
        centers.append(c2w[:, 3])
        rots.append(c2w[:, :3])
    return (np.array(pts), np.array(cols, dtype=np.uint8), np.array(centers), np.array(rots))


def build_collision_mesh(points: np.ndarray, centers: np.ndarray, voxel: float = 0.12,
                         path_clearance_m: float = 0.45) -> trimesh.Trimesh:
    """Voxel boxes at obstacle points (0.16 m..1.8 m) + a floor patch under the camera path.

    The capture walked the camera path, so it is free space by definition: obstacle points
    within path_clearance_m (horizontally) of any camera centre are SfM noise and are dropped.
    """
    from scipy.spatial import cKDTree

    parts = []
    obstacle = points[(points[:, 1] > 0.16) & (points[:, 1] < 1.8)]
    if len(obstacle) and len(centers):
        d, _ = cKDTree(centers[:, [0, 2]]).query(obstacle[:, [0, 2]])
        obstacle = obstacle[d > path_clearance_m]
    if len(obstacle):
        vox = np.unique(np.floor(obstacle / voxel).astype(np.int32), axis=0)
        for v in vox[:20000]:
            b = trimesh.creation.box(extents=[voxel] * 3)
            b.apply_translation((v + 0.5) * voxel)
            parts.append(b)
    lo = np.minimum(centers.min(0), points.min(0)) - 1.0
    hi = np.maximum(centers.max(0), points.max(0)) + 1.0
    floor = trimesh.creation.box(extents=[hi[0] - lo[0], 0.02, hi[2] - lo[2]])
    floor.apply_translation([(hi[0] + lo[0]) / 2, -0.01, (hi[2] + lo[2]) / 2])
    parts.append(floor)
    return trimesh.util.concatenate(parts)


def reconstruct(video_path: Path, scene_id: str, min_parallax_px: float = 8.0,
                min_sharpness: float = 40.0, camera_height_m: float = 1.45, window: int = 10) -> dict:
    """window: at most one keyframe per this many video frames (the sharpest of them).
    Short clips (stock footage, ~20 s) need a smaller window to get enough views."""
    scene_dir = DATA_ROOT / scene_id
    art = scene_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    kf_dir = scene_dir / "keyframes"

    logger.info("Stage 1/4 keyframe selection")
    if kf_dir.exists():  # stale frames from an earlier build would leak into SfM
        import shutil
        shutil.rmtree(kf_dir)
    kfs = SmartFrameSelector(min_sharpness=min_sharpness, min_parallax_px=min_parallax_px,
                             window_size=window).process_video(
        Path(video_path), kf_dir)
    logger.info("Stage 2/4 SfM (pycolmap)")
    rec, n_models, best_key = run_sfm(kf_dir, scene_dir / "colmap")
    pts, cols, centers, rots = extract_geometry(rec)

    logger.info("Stage 3/4 gravity + metric alignment")
    aligner = MetricGravityAligner(camera_height_m)
    sim3, scale = aligner.compute_canonical_sim3(trimesh.Trimesh(vertices=pts), centers, rots)
    pts_a = (sim3[:3, :3] @ pts.T).T + sim3[:3, 3]
    cen_a = (sim3[:3, :3] @ centers.T).T + sim3[:3, 3]
    np.save(art / "sim3_alignment.npy", sim3)

    # drop far outliers so the viewer frames the room, not stray points
    keep = np.linalg.norm(pts_a - np.median(pts_a, axis=0), axis=1) < np.percentile(
        np.linalg.norm(pts_a - np.median(pts_a, axis=0), axis=1), 97)
    pts_a, cols = pts_a[keep], cols[keep]

    logger.info("Stage 4/4 export")
    trimesh.PointCloud(pts_a, colors=cols).export(str(art / "points.ply"))
    build_collision_mesh(pts_a, cen_a).export(str(art / "collision_mesh.glb"))
    # camera viewing directions (aligned frame) let the viewer start inside, facing the capture
    R = sim3[:3, :3] / np.cbrt(np.linalg.det(sim3[:3, :3]))
    fwd_a = np.einsum("ij,njk->nik", R, rots)[:, :, 2]
    (art / "cameras.json").write_text(json.dumps({
        "centers": np.round(cen_a, 3).tolist(), "forwards": np.round(fwd_a, 3).tolist()}))
    yaw = np.degrees(np.arctan2(fwd_a[:, 0], fwd_a[:, 2])) % 360
    yaw_gap = float(np.diff(np.sort(np.r_[yaw, yaw.min() + 360])).max())
    report = {
        "scene_id": scene_id, "keyframes": len(kfs), "models": n_models,
        "registered_images": int(rec.num_reg_images()), "points": int(len(pts_a)),
        "mean_reprojection_px": round(float(rec.compute_mean_reprojection_error()), 3),
        "largest_view_direction_gap_deg": round(yaw_gap, 1),  # < ~45 means you can turn around
        "metric_scale": float(scale), "assumed_camera_height_m": camera_height_m,
        "best_model_dir": str(scene_dir / "colmap" / str(best_key)), "keyframes_dir": str(kf_dir),
    }
    (art / "scene_report.json").write_text(json.dumps(report, indent=2))
    frames = sorted(kf_dir.glob("*.jpg"))
    if frames:
        import shutil
        shutil.copy(frames[len(frames) // 2], art / "reference.jpg")
    return report
