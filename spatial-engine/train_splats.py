#!/usr/bin/env python
"""python train_splats.py <colmap_model_dir> <image_dir> <scene_id> [--iters 2500] [--width 320] [--out x.ply] [--no-fill]
                        [--engine gsplat]

Trains gaussians in the gravity-aligned metric frame and writes artifacts/visual_splats.ply.
--engine gsplat uses gsplat's CUDA rasteriser (needs the CUDA toolkit and C++ build tools to
compile once): train on full-resolution keyframes for 30k steps, e.g.
    python train_splats.py data/scenes/x/colmap/0 data/scenes/x/keyframes x --engine gsplat --iters 30000 --width 0
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pycolmap
import trimesh

from backend.config import DATA_ROOT
from backend.reconstruction.gravity_aligner import MetricGravityAligner
from backend.reconstruction.pycolmap_pipeline import extract_geometry
from backend.reconstruction.multi_model import gather, load_views, refresh_capture
from backend.reconstruction.splat_trainer import train


def run(model_dir, image_dir, scene_id, iters=2500, width=320, device="cuda", out=None, fill=True, engine="torch",
        cap=500_000):
    rec = pycolmap.Reconstruction(str(model_dir))
    pts, cols, cen, rots = extract_geometry(rec)
    sim3 = scene_frame(scene_id, pts, cen, rots)
    # frames and points of sibling SfM models that align to this one (rooms the main model missed)
    frames, pts, cols, merged = gather(model_dir, log=lambda m: print(m, flush=True))
    views, K = load_views(frames, Path(image_dir), sim3, width)
    xyz = (sim3[:3, :3] @ pts.T).T + sim3[:3, 3]
    if merged["merged"] and not out:  # capture spots, map and walking cover the added rooms too
        refresh_capture(DATA_ROOT / scene_id / "artifacts", frames, pts, cols, sim3)
        print("camera path, points and collision mesh now include the merged frames", flush=True)
    out = Path(out) if out else DATA_ROOT / scene_id / "artifacts" / "visual_splats.ply"
    print("views", len(views), "points", len(xyz), "K", K, flush=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if engine == "gsplat":
        from backend.reconstruction.gsplat_trainer import train as gs_train
        psnr = gs_train(views, K, xyz, cols, out, iters=iters, device=device, log=log, fill=fill, cap=cap)
    else:
        psnr = train(views, K, xyz, cols, out, iters=iters, device=device, log=log, fill=fill)
    print("wrote", out, flush=True)
    make_ksplat(out)
    return psnr


def scene_frame(scene_id, pts, cen, rots):
    """The gravity-aligned metric frame the scene's build saved (its mesh, camera path and
    points are in it). Recomputing it from the model reloaded from disk can land a few degrees
    and centimetres away, which would offset the splats from everything else."""
    saved = DATA_ROOT / scene_id / "artifacts" / "sim3_alignment.npy"
    if saved.exists():
        return np.load(saved)
    return MetricGravityAligner(1.45).compute_canonical_sim3(trimesh.Trimesh(vertices=pts), cen, rots)[0]


def make_ksplat(ply):
    """Write the compact .ksplat the viewer loads first (needs Node and the frontend's packages)."""
    frontend = Path(__file__).parent / "frontend"
    node = shutil.which("node")
    if not node or not (frontend / "node_modules").exists():
        print("[ksplat] skipped: run `npm install` in frontend/ and `npm run ksplat` for faster loading", flush=True)
        return
    r = subprocess.run([node, "scripts/make-ksplat.mjs", str(Path(ply).resolve()), "--force"], cwd=frontend,
                       capture_output=True, text=True)
    print("[ksplat]", (r.stdout or r.stderr).strip(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images")
    ap.add_argument("scene_id")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--width", type=int, default=320, help="training image width (0: keyframe size)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", choices=("torch", "gsplat"), default="torch",
                    help="torch: pure PyTorch, runs anywhere; gsplat: CUDA, fast enough for full resolution")
    ap.add_argument("--out", help="output .ply (default: the scene's artifacts/visual_splats.ply)")
    ap.add_argument("--no-fill", action="store_true", help="do not guess gaussians for gaps in the SfM cloud")
    ap.add_argument("--cap", type=int, default=500_000, help="gsplat: most gaussians during training")
    a = ap.parse_args()
    run(a.model, a.images, a.scene_id, a.iters, a.width, a.device, a.out, not a.no_fill, a.engine, a.cap)
