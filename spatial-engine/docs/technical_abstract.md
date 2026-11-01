# Namma Space, Round 1 - Technical Abstract

## 1. Problem and approach
Turn an ordinary phone walkthrough video of an indoor space into a browsable 3D model with free roaming, collision-free navigation and point-of-interest (POI) search, using only open-source components that allow commercial use.

The engine keeps two aligned representations of the space:

- **Visual layer:** a 3D Gaussian Splat (or, when splats are unavailable, a coloured point cloud) for appearance.
- **Geometric layer:** an invisible collision mesh in the same metric, Y-up frame. It drives BVH raycasting, a 2.5D navigation grid with a Euclidean distance field, and path planning.

## 2. Standard Operating Procedure (capture)
1. Hold the phone upright at chest height (about 1.45 m). Do not use the ultrawide lens.
2. Walk slowly (under 0.5 m/s) along a loop, keeping every area visible from several angles. Pan, do not spin on the spot.
3. Keep lighting steady; avoid mirrors, glass and blank walls in the frame centre.
4. Record 1-3 minutes at 1080p or higher, 30 fps. Optional: record a reference length (e.g. a door, 2.0 m) for metric scale.

## 3. Pipeline
| Stage | Method | Module |
|---|---|---|
| Keyframes | Laplacian-variance blur rejection, Lucas-Kanade parallax gating | `ingestion/frame_selector.py` |
| Structure from motion | COLMAP (BSD) through `pycolmap`: SIFT, sequential matching, incremental mapping; best model chosen by triangulated points | `reconstruction/pycolmap_pipeline.py` |
| Alignment | RANSAC floor plane, rotation to Y-up, floor at Y=0, metric scale from camera height (or VIO poses when supplied) | `reconstruction/gravity_aligner.py` |
| Appearance | Gaussian splatting trained in the aligned frame (pure PyTorch trainer, no compiled extensions) | `reconstruction/splat_trainer.py` |
| Collision | Voxel-surface mesh plus floor patch, exported as GLB | `pycolmap_pipeline.py` |
| Navigation | ESDF grid from the mesh, Theta* any-angle search, collision-checked B-spline | `navigation/pathplanner.py` |
| Semantics | POI database (label + 3D box); voxel CLIP-fusion code is present but not auto-labelled | `semantics/` |
| Serving | FastAPI backend and a Three.js viewer (orbit/roam, double-click to navigate, POI search) | `backend/main.py`, `frontend/` |

All libraries (COLMAP, OpenCV, SciPy, trimesh, PyTorch, Three.js, gaussian-splats-3d) are permissively licensed. SuperPoint/SuperGlue are deliberately not used.

## 4. Results so far (honest status)
- Verified end to end on a public 126 s indoor video (Wikimedia Commons, CC BY-SA 4.0): 506 frames at 4 fps -> sparse model of 63 registered images and 7.2k points, rendered in the web viewer.
- The same video fragments into about 10 SfM models, because of weak overlap and low resolution (360p). Continuous raw phone video is expected to register far better.
- Metric scale is assumed from camera height, not measured. Dimensions are indicative only.
- Unit and API tests cover alignment, path planning, POI search and the HTTP endpoints.

## 5. Known limitations and next steps
- Train splats on a GPU at higher resolution and with more iterations; compare against the point-cloud fallback.
- Automatic POI labelling (open-vocabulary detector + CLIP fusion) with a commercially usable model.
- Loop closure and multi-room stitching; scale calibration from a reference object.
