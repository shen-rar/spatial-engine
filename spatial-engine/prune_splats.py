#!/usr/bin/env python
"""python prune_splats.py <scene_id> [--keep 0.5] [--finetune 5000] [--dry-run]

Makes a scene lighter to draw (smoother walking in the browser, smaller download): drops the
gaussians that add least to the training images, then briefly fine-tunes the rest so they cover
for them (backend/reconstruction/splat_pruner.py). Prints held-out PSNR/SSIM (every 8th capture,
never trained on) before and after, and writes the scene's visual_splats.ply and .ksplat in place.
Needs gsplat (see train_splats.py --engine gsplat), the scene's COLMAP model and keyframes.
"""
import argparse
import json
from pathlib import Path

import pycolmap
import torch

from backend.config import DATA_ROOT
from backend.reconstruction.gsplat_trainer import _import_gsplat
from backend.reconstruction.pycolmap_pipeline import extract_geometry
from backend.reconstruction.splat_pruner import evaluate, finetune, importance, read_ply, to_gaussians, write_ply
from backend.reconstruction.multi_model import gather, load_views
from train_splats import make_ksplat, scene_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_id")
    ap.add_argument("--keep", type=float, default=0.5, help="share of gaussians to keep (0-1)")
    ap.add_argument("--finetune", type=int, default=5000, help="fine-tuning steps after pruning (0: none)")
    ap.add_argument("--width", type=int, default=0, help="render width (0: keyframe size)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    art = DATA_ROOT / a.scene_id / "artifacts"
    report = json.loads((art / "scene_report.json").read_text(encoding="utf-8"))
    rasterization, _ = _import_gsplat()
    rec = pycolmap.Reconstruction(report["best_model_dir"])
    pts, _, cen, rots = extract_geometry(rec)
    frames, _, _, _ = gather(report["best_model_dir"], log=lambda m: print(m, flush=True))  # same views as training
    views, K = load_views(frames, Path(report["keyframes_dir"]), scene_frame(a.scene_id, pts, cen, rots), a.width)
    H, W = views[0]["img"].shape[:2]
    hold = set(range(0, len(views), 8))
    train = [v for i, v in enumerate(views) if i not in hold]
    held = [v for i, v in enumerate(views) if i in hold]

    names, data = read_ply(art / "visual_splats.ply")
    gs = to_gaussians(names, data)
    n = len(data)
    p0, s0 = evaluate(gs, held, K, rasterization, W, H)
    print(f"{a.scene_id}: {n} gaussians, held-out PSNR {p0:.2f} dB SSIM {s0:.3f} at {W}px", flush=True)

    score = importance(gs, train, K, rasterization, W, H)
    idx = torch.argsort(score, descending=True)[: max(1, int(n * a.keep))].sort().values.cpu().numpy()
    del gs
    kept = data[idx]
    p1, s1 = evaluate(to_gaussians(names, kept), held, K, rasterization, W, H)
    print(f"pruned to {len(idx)} ({a.keep:.0%}): PSNR {p1:.2f} ({p1 - p0:+.2f}) SSIM {s1:.3f}", flush=True)
    if a.finetune:
        kept = finetune(names, kept, train, K, rasterization, W, H, a.finetune, log=lambda m: print(m, flush=True))
        p2, s2 = evaluate(to_gaussians(names, kept), held, K, rasterization, W, H)
        print(f"after fine-tuning: PSNR {p2:.2f} ({p2 - p0:+.2f}) SSIM {s2:.3f}", flush=True)
    if not a.dry_run:
        write_ply(art / "visual_splats.ply", names, kept)
        print("wrote", art / "visual_splats.ply", flush=True)
        make_ksplat(art / "visual_splats.ply")


if __name__ == "__main__":
    main()
