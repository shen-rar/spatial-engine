#!/usr/bin/env python
"""
One command from raw video to a viewable scene:

    python build_scene.py my_video.mp4 my_scene
    python build_scene.py my_video.mp4 my_scene --iters 4000   # longer, better splats
    python build_scene.py my_video.mp4 my_scene --no-splats    # point cloud + mesh only
    python build_scene.py clip.mp4 my_scene --window 5 --parallax 3   # short (~20 s) clips

Steps: keyframes -> COLMAP SfM -> gravity/metric alignment -> collision mesh
       -> (optional) gaussian splat training -> scene_report.json
Then:  python -m uvicorn backend.main:app --port 8000   and   npm run dev (frontend)
"""
import argparse
import json
import logging
import time

from pathlib import Path

from backend.reconstruction.pycolmap_pipeline import reconstruct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("scene_id")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--width", type=int, default=320, help="training image width (0: keyframe size)")
    ap.add_argument("--engine", choices=("torch", "gsplat"), default="torch", help="splat trainer (see train_splats.py)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parallax", type=float, default=8.0)
    ap.add_argument("--sharp", type=float, default=40.0)
    ap.add_argument("--window", type=int, default=10,
                    help="at most one keyframe per N video frames; use ~5 for clips under 30 s")
    ap.add_argument("--no-splats", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    report = reconstruct(a.video, a.scene_id, a.parallax, a.sharp, window=a.window)
    print(f"[build] geometry done in {time.time() - t0:.0f}s: {report['registered_images']} images, {report['points']} points")

    if not a.no_splats:
        if report["registered_images"] < 10:
            print("[build] too few registered images for splat training - skipping. Capture with more overlap.")
        else:
            from train_splats import run as train_run
            train_run(report["best_model_dir"], report["keyframes_dir"], a.scene_id, a.iters, a.width, a.device,
                      engine=a.engine)

    print(json.dumps({"scene": a.scene_id, "total_s": round(time.time() - t0)}, indent=2))
    print(f"[build] open: http://localhost:5173/?scene={a.scene_id}")


if __name__ == "__main__":
    main()
