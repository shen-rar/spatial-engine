"""
Use every registered frame, not just the main SfM model's.

COLMAP often splits a walkthrough into several models: a fast turn or a blurry stretch breaks the
chain, and the frames after it start a model of their own. The scene is built from the largest
model, so whole rooms filmed in the smaller ones have no training images and show as empty or
foggy space. Other models that share frames with the main one (directly, or through another
model) are brought into its frame with a similarity transform fitted on the shared camera
centres, and their extra frames and points are added. The main model's frame is unchanged, so
the mesh, camera path and blueprint built from it stay valid.
"""
from pathlib import Path

import numpy as np


def _umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / a.var(0).sum())
    return s, R, md - s * R @ ms


def _poses(rec):
    """name -> (camera, c2w 3x4) for every image of a model."""
    return {img.name: (rec.cameras[img.camera_id], img.cam_from_world().inverse().matrix())
            for img in rec.images.values()}


def gather(model_dir, min_shared=6, max_err=0.05, max_rot_deg=3.0, log=print):
    """Main model at model_dir plus every sibling model that aligns to it. Returns
    (frames {name: (camera, c2w in the main model's frame)}, points (N,3), colours (N,3), report)."""
    import pycolmap

    from backend.reconstruction.pycolmap_pipeline import extract_geometry

    model_dir = Path(model_dir)
    main = pycolmap.Reconstruction(str(model_dir))
    frames = _poses(main)
    pts, cols, _, _ = extract_geometry(main)
    pts, cols = [pts], [cols]
    others = {p.name: pycolmap.Reconstruction(str(p)) for p in sorted(model_dir.parent.iterdir())
              if p.is_dir() and p != model_dir and (p / "images.bin").exists()}
    report = {"main": model_dir.name, "main_frames": len(frames), "merged": {}}
    # keep sweeping: a model may only connect to the main one through another merged model
    progress = True
    while progress and others:
        progress = False
        for key, rec in list(others.items()):
            poses = _poses(rec)
            shared = sorted(set(poses) & set(frames))
            if len(shared) < min_shared:
                continue
            src = np.array([poses[n][1][:, 3] for n in shared])
            dst = np.array([frames[n][1][:, 3] for n in shared])
            s, R, t = _umeyama(src, dst)
            span = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean()) or 1.0
            err = float(np.linalg.norm(src @ (s * R).T + t - dst, axis=1).mean()) / span
            # the shared cameras must also agree on where they look
            rot = max(np.degrees(np.arccos(np.clip((np.trace(frames[n][1][:, :3].T @ R @ poses[n][1][:, :3]) - 1) / 2, -1, 1)))
                      for n in shared)
            del others[key]
            if err > max_err or rot > max_rot_deg:
                log(f"model {key}: {len(shared)} shared frames but no clean fit (centre error {err:.1%} of the "
                    f"walk, orientation {rot:.1f} deg); left out")
                continue
            new = [n for n in poses if n not in frames]
            for n in new:
                cam, c2w = poses[n]
                frames[n] = (cam, np.hstack([R @ c2w[:, :3], (s * R @ c2w[:, 3] + t)[:, None]]))
            p, c, _, _ = extract_geometry(rec)
            pts.append(p @ (s * R).T + t)
            cols.append(c)
            report["merged"][key] = {"shared": len(shared), "new_frames": len(new), "centre_error": round(err, 4),
                                     "orientation_deg": round(rot, 2)}
            log(f"model {key}: +{len(new)} frames via {len(shared)} shared (centre error {err:.1%}, "
                f"orientation {rot:.1f} deg)")
            progress = True
    return frames, np.concatenate(pts), np.concatenate(cols), report


def refresh_capture(art: Path, frames, pts, cols, sim3):
    """Rewrite the scene's camera path, points and collision mesh from the gathered frames, so
    capture spots, the map and walking cover the rooms the merged models added. cameras.json
    lists frames in name order (as the pipeline does) and now names them."""
    import json

    import trimesh

    from backend.reconstruction.pycolmap_pipeline import build_collision_mesh

    names = sorted(frames)
    S = sim3[:3, :3]
    R = S / np.cbrt(np.linalg.det(S))
    cen = np.array([S @ frames[n][1][:, 3] + sim3[:3, 3] for n in names])
    fwd = np.array([R @ frames[n][1][:, 2] for n in names])
    pa = pts @ S.T + sim3[:3, 3]
    d = np.linalg.norm(pa - np.median(pa, axis=0), axis=1)
    keep = d < np.percentile(d, 97)  # as the pipeline: drop far outliers
    trimesh.PointCloud(pa[keep], colors=cols[keep]).export(str(art / "points.ply"))
    build_collision_mesh(pa[keep], cen).export(str(art / "collision_mesh.glb"))
    (art / "cameras.json").write_text(json.dumps({
        "centers": np.round(cen, 3).tolist(), "forwards": np.round(fwd, 3).tolist(), "names": names}))


def load_views(frames, image_dir: Path, sim3: np.ndarray, width: int, ref_camera=None):
    """Training views from gathered frames, all resampled into one pinhole camera (the main
    model's, or ref_camera): each frame is undistorted with its own model's intrinsics and
    re-projected with the reference ones, so a single K fits every view. Sorted by name."""
    import cv2
    import torch

    cams = [c for c, _ in frames.values()]
    ref = ref_camera or max(cams, key=lambda c: sum(o is c for o in cams))
    fr, cxr, cyr = ref.params[0], ref.params[1], ref.params[2]
    Kref = np.array([[fr, 0, cxr], [0, fr, cyr], [0, 0, 1]], dtype=np.float64)
    S = sim3[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(S))))
    Rg = S / s
    views, K = [], None
    for name in sorted(frames):
        cam, c2w = frames[name]
        raw = cv2.imread(str(image_dir / name))
        if raw is None:
            continue
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        k1 = cam.params[3] if len(cam.params) > 3 else 0.0
        Km = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        und = cv2.undistort(raw, Km, np.array([k1, 0, 0, 0], dtype=np.float64), newCameraMatrix=Kref)
        scale = width / und.shape[1] if width else 1.0
        if scale != 1.0:
            und = cv2.resize(und, (width, int(round(und.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        K = (fr * scale, fr * scale, cxr * scale, cyr * scale)
        R_c2w = Rg @ c2w[:, :3]
        center = S @ c2w[:, 3] + sim3[:3, 3]
        w2c = np.eye(4)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -R_c2w.T @ center
        views.append({"name": name, "img": torch.from_numpy(cv2.cvtColor(und, cv2.COLOR_BGR2RGB)).float() / 255.0,
                      "w2c": torch.from_numpy(w2c).float()})
    return views, K
