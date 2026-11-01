#!/usr/bin/env python
"""Usage: python run_pipeline.py <video> <scene_id> [--parallax 8] [--sharp 40]"""
import argparse, json, logging
from pathlib import Path
from backend.reconstruction.pycolmap_pipeline import reconstruct

logging.basicConfig(level=logging.INFO)
ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("scene_id")
ap.add_argument("--parallax", type=float, default=8.0)
ap.add_argument("--sharp", type=float, default=40.0)
a = ap.parse_args()
print(json.dumps(reconstruct(Path(a.video), a.scene_id, a.parallax, a.sharp), indent=2))
