# Spatial Engine - Namma Space Round 1

Video -> gravity-aligned, metric 3D scene -> web viewer with navigation and POI search.

## Pipeline (SOP)
1. **Ingest** - `backend/ingestion/frame_selector.py`: blur rejection (Laplacian variance) + optical-flow parallax gating -> keyframes.
2. **SfM** - `backend/reconstruction/pycolmap_pipeline.py`: COLMAP (BSD) via `pycolmap`: SIFT features, sequential matching, incremental mapping.
3. **Align** - `gravity_aligner.py`: RANSAC floor plane -> Y-up, floor at Y=0; metric scale from assumed camera height (1.45 m) or ARKit poses if supplied.
4. **Export** - `points.ply` (visual layer), `collision_mesh.glb` (invisible BVH layer), `cameras.json`, `scene_report.json`. A 3DGS `visual_splats.ksplat` placed in `artifacts/` is picked up automatically by the viewer.
5. **Serve** - FastAPI (nav grid + Theta* path planning + POI search) and a Three.js viewer.

All dependencies are open source / commercially usable (no SuperPoint/SuperGlue).

## Setup
```bash
pip install numpy scipy opencv-python-headless trimesh fastapi "uvicorn[standard]" pydantic pycolmap pytest httpx
cd frontend && npm install && cd ..
```

## One-command build
```bash
python build_scene.py my_video.mp4 my_scene              # geometry + splats (GPU)
python build_scene.py my_video.mp4 my_scene --no-splats  # geometry only, no GPU needed
python build_scene.py clip.mp4 my_scene --window 5 --parallax 3 --iters 7000 --width 480   # short clips (~20 s)
python build_blueprint.py my_scene_1 --from my_scene --name "My scene · blueprint"         # its blueprint twin
```
Then start the backend and frontend as below and open `/?scene=my_scene`. A blueprint scene
built `--from` a photoreal one is its twin: the viewer lists the house once and flips between
the two with the Photoreal / Blueprint toggle.

### Shipped scenes
| House | Photoreal | Blueprint | Source video |
|---|---|---|---|
| House | `house` | `house_1` | own capture |
| Beam House | `beam_house` | `beam_house_1` | *Video of a House Interior*, Kindel Media, Pexels license (`data/videos/house_interior_7578546.mp4`) |
| Open-Plan House | `open_plan` | `open_plan_1` | *Modern Open-Concept Luxury Living Room and Kitchen Design*, Ethan Raven, Pexels license (`data/videos/open_plan_31617700.mp4`) |

Open-Plan House covers the living room and kitchen; the white lounge and den at the end of the
clip pass too fast to register. The viewer opens a photoreal scene at the middle capture
viewpoint, or at `"start_view": <index>` from its `scene.json` (Open-Plan uses 4, the living
room, because the middle of its walk is a close-up of the kitchen counter).

## Run
```bash
python create_demo_scene.py                       # test fixture (data/scenes/demo_office, hidden from the viewer)
python run_pipeline.py my_video.mp4 my_scene      # reconstruct your own video
python -m uvicorn backend.main:app --port 8000    # backend
cd frontend && npm run dev                        # viewer: http://localhost:5173/?scene=my_scene
```
Fastest for viewing (no dev server): `cd frontend && npm run build`, then the backend alone
serves the bundled site at http://localhost:8000/?scene=my_scene.

Public hosting without a server (live at https://spatial-engine.spatial-engine.workers.dev):
`python export_static.py site` writes the built viewer, the files each scene loads and the
backend's read-only answers (scene list, floor-plan map, calibration) into `site/`, with
`_headers` for cross-origin isolation and `_redirects` mapping the API paths onto the files.
`npx wrangler deploy` uploads it (see `wrangler.jsonc`). Without the backend, routes are planned
in the browser (`GridPlanner.js`) along the capture walk and the open floor near it, and a
calibration applies to that visit only.
On Windows set `PYTHONUTF8=1` before running tests. Tests: `python -m pytest tests -v`.

## Viewer (Field System)
Layout modelled on the renderify field viewer: plan map, real-frame panel, enter-the-space gate and a sandbox.

- **Plan map (bottom right / top right):** live map from the backend's per-cell status. Open, blocked (obstacle or too close to a wall), unseen (no floor observed), capture path, route and placed props. Click the map to route to that spot.
- **Sandbox (G):** place cone, barrier, crate or sign (keys 1-4), click to place, R rotates, Delete removes the prop under the reticle, T routes to the last prop. Placement is checked against the map: blocked and unseen cells refuse with a reason. Layouts persist per scene in this browser and the tray's "Share" copies them as a link (`layout=`).
- **Real frame:** a captured frame saved by the build as `artifacts/reference.jpg`.
- **Status stamp:** LIVE when splats or a point cloud are loaded, OFFLINE otherwise.
- **Model switcher (top of the scene panel):** lists each house the backend serves (`GET /api/v1/scenes`) once and reloads into the one you pick, keeping the current view style.
- **Photoreal / Blueprint toggle (next to the title):** flips the house between its gaussian splats and its line blueprint.
- **Object cards (blueprint):** hover a furniture piece for its name, click it to highlight it and
  open a card with its photo from the capture, name and dimensions (metres and feet; values the
  capture could not see are marked *typical*). The orbit centres on the piece: drag or press
  *Orbit 360°* to turn around it. Esc closes the card. *Find in scene* lists every piece of the
  blueprint (grouped: furniture, lighting, décor, walls and windows, kitchen and storage) and
  filters as you type; picking a row (or Enter for the top match) selects it just like clicking it. House has the most pieces (17)
  and the fullest 360° coverage, so it is the best scene for this.

### Speed
- Splats load from `visual_splats.ksplat` (16-bit, ~40% of the PLY, streamed and shown as it
  arrives). Training writes it automatically; for older scenes run `cd frontend && npm run ksplat`.
- The scene, its collision mesh, camera path and navigation grid load in parallel; the
  navigation grid is cached per scene on the backend.
- The view is drawn only while something changes; a still view uses no GPU. While moving, the
  render resolution adapts to hold ~50 fps and returns to full sharpness when the view stops.
- The backend sends cross-origin isolation headers so the splat sorter shares memory with its
  worker, gzips JSON and scripts, and lets browsers revalidate scene files by ETag.
- Splats are fill bound: frame time scales with screen pixels. The status bar shows
  "GPU: integrated (slower)" when the browser renders on an integrated GPU (hover for the fix).
- On laptops with two GPUs, set the browser to *High performance* in Windows Settings > System >
  Display > Graphics: splats on an integrated GPU are several times slower.

| Action | Control |
|---|---|
| Turn around in place (360°) | drag; the viewer starts inside the room at a capture spot, at eye height |
| Automatic 360° look-around | **O** |
| Previous / next capture spot | **[** / **]** (spots where the camera actually stood, so splats look sharpest) |
| Orbit / pan / zoom the whole scene | **M** (overview), then mouse |
| Walk first person (wall collision) | **F**, then WASD + mouse, Shift sprints, Esc exits |
| Auto-navigate to a spot | double-click the floor |
| POI search + fly-to | type a label, e.g. "espresso machine" |
| Plan (top-down) view | **M** |
| Toggle points / mesh / blueprint lines | **P** |
| Measure distance | **U**, click two points; the first measurement asks for a real length to calibrate the scale |

Set `VITE_API_BASE_URL` for a remote backend, or `API_PROXY_TARGET` to point the dev server's proxy at a backend other than `localhost:8000`. Viewing works without the backend; navigation and search need it.

## Training splats (GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python train_splats.py <colmap_model_dir> <image_dir> <scene_id> --iters 7000 --width 480   # house: ~2 h on an 8 GB laptop GPU
```
Writes `artifacts/visual_splats.ply` (or `--out`), which the viewer prefers over the point cloud. The default trainer is pure PyTorch (no compiler needed): all screen tiles are rasterised in batched GPU passes, with L1 + SSIM loss, opacity resets against floaters, per-frame exposure correction, SH degree 1 colour, and guessed fill gaussians for gaps in the SfM cloud (`--no-fill` turns them off).

### Sharper: gsplat at full resolution
```bash
pip install gsplat==1.5.3
python train_splats.py data/scenes/beam_house/colmap/1 data/scenes/beam_house/keyframes beam_house --engine gsplat --iters 30000 --width 0
python build_scene.py my_video.mp4 my_scene --engine gsplat --iters 30000 --width 0
```
`--engine gsplat` trains with gsplat's CUDA rasteriser and MCMC densification (up to 500k
gaussians), on the full keyframes (`--width 0`, 1600 px) instead of 480 px. Same output files,
so the viewer and blueprints need no changes. ~45 min per house on an RTX 5060 Laptop; held-out
views (every 8th) at 1600 px:

| House | PyTorch trainer (480 px, 7k steps) | gsplat (1600 px, 30k steps) |
|---|---|---|
| Beam House | 23.0 dB / SSIM 0.895 | 30.1 dB / SSIM 0.969 |
| Open-Plan House | 17.0 dB / SSIM 0.792 | 21.4 dB / SSIM 0.878 |

Re-run `build_blueprint.py ... --measure` after retraining: furniture is measured from the splats.

gsplat compiles its kernels once on first use (~80 s). It needs the CUDA toolkit matching torch
(12.8 for RTX 50-series) and, on Windows, the Visual Studio 2022 C++ build tools; run from a
shell that has called `vcvars64.bat` and set `CUDA_HOME`:
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set TORCH_CUDA_ARCH_LIST=12.0
```
gsplat 1.5.3 does not build on Windows with torch 2.11 as shipped; `gsplat_trainer` patches the
build (MSVC flags, and `win_cuda_fix.h` for the `small` macro from `windows.h`).

Sharper training only helps where the video looked: directions the camera never faced stay
empty or foggy, whatever the renderer.

Training uses the frame the build saved (`artifacts/sim3_alignment.npy`, the frame of the mesh,
camera path and points). Recomputing it from the COLMAP model on disk lands a few degrees and
centimetres away, which offset splats from the walk mesh and blurred the capture-spot views.

### Every registered frame, not just the main model's
COLMAP can split a walkthrough into several models (a fast turn breaks the chain). Training uses
the main model plus every sibling model in `colmap/` that shares at least 6 frames with it
(directly or through another sibling): a similarity transform fitted on the shared camera
centres brings their frames and points into the main model's frame (rejected if the fit is off
by more than 5% of the walk or 3°), and each frame is resampled into the main camera's
intrinsics (`backend/reconstruction/multi_model.py`). When frames were added, training also
rewrites the camera path (now with frame names), points and collision mesh, so capture spots, the
map and walking reach the added rooms; re-run `build_blueprint.py ... --measure` afterwards.

House was split 99 / 44 / 25 / 3 frames; merging adds 29 frames, the whole living and dining
hall. Held-out views in that hall went from 14.7 dB (smears and streaks) to 27.2 dB.
`python train_splats.py data/scenes/house/colmap/1 data/scenes/house/keyframes house --engine gsplat --iters 40000 --width 0 --cap 600000`,
then `prune_splats.py house --keep 0.7`.

### Lighter: prune after training
```bash
python prune_splats.py beam_house --keep 0.55     # same gsplat shell as above
```
Scores every gaussian by the alpha it adds to the training images, keeps the top 55%, then
fine-tunes them for 5k steps so they cover for the dropped ones (LightGaussian-style). The
browser draws far fewer, smaller splats; held-out quality goes up (fewer floaters). Writes the
`.ply` and `.ksplat` in place; re-run `build_blueprint.py ... --measure` afterwards.

| House | Splats | Download | Held-out PSNR / SSIM | Frame on an Intel UHD (1x / 1.5x) |
|---|---|---|---|---|
| Beam House | 186k → 111k | 7.9 → 4.7 MB | 30.1 → 32.7 dB / 0.969 → 0.969 | 18.6 / 27.9 → 4.9 / 10.6 ms |
| Open-Plan House | 340k → 198k | 14.3 → 8.3 MB | 21.4 → 22.4 dB / 0.878 → 0.866 | 10.1 / ~20 → 7.2 / 12.1 ms |
| House (gsplat, 128 merged frames, 720 px, keep 70%) | 215k → 177k | 9.0 → 7.4 MB | 24.2 → 25.3 dB / 0.908 → 0.910 | 56 / 94 → 8.9 / 12.5 ms |

Frame times are the GPU cost of drawing along the capture walk (`renderer.render` plus a 1-pixel
read-back), 1024×768 at the given pixel ratio.

## Blueprint model (lines only)
```bash
python build_blueprint.py house_1 --from house --name "House · blueprint"
```
Builds a second scene that draws the capture as a 3D blueprint: colours and object detail are
dropped, leaving the floor outline extruded to the local ceiling height, interior walls, and
furniture as plain boxes, all snapped to the dominant wall direction. It is measured from the
source scene's splats and SfM points and written to `artifacts/blueprint.json`; the scene's
`scene.json` (`"style": "blueprint"`) tells the viewer to draw lines instead of splats. Walls
the capture saw poorly (white walls, glass) may be missing or approximate.

Furniture is drawn as recognisable line pieces (L sofa with back and cushions, tub armchairs,
round coffee table, dining table with chairs, pool table with pockets, kitchen island with hob
and hood, bar with sink, staircase with steps, TV on its console, terrace chairs and table).
Each piece is marked as a box in a captured frame in `furniture_annotations.json`; then
```bash
python build_blueprint.py house_1 --from house --measure
```
(add `"width_m"` to an annotation when the piece is only seen end-on, e.g. a long table receding
from the camera) reads each piece's position and width from the trained splats' depth (needs the source's COLMAP
model, keyframes and a GPU) into `artifacts/furniture.json`. Depth front-to-back is the type's
typical size, and overlapping pieces are nudged apart. Without annotations, furniture-sized
point clusters are drawn as plain boxes. Only frames that registered in the source's COLMAP
model can be annotated; a piece whose box has too few splats behind it is skipped.

Inside a marked box the piece is the depth layer that fills the box best across and top to
bottom, so a coffee table in front of a sofa, or stools in front of an island, are not mistaken
for it; with several marks, the clearest view sets the position. Long pieces (sofas, islands,
tables, counters) take their direction from the spread of their own points, snapped to the walls
only within 15°, so a sofa set at an angle is drawn at that angle; stools may tuck under counters.
Last, each piece is fine-tuned (within 0.5 m, over its candidate orientations) so its 3D box, seen
from every frame it was marked in, covers the marked box: the photoreal model reproduces those
frames at their capture spots, so this is what lines the blueprint up with it. Placement against
the marked boxes (mean IoU of the projected piece) went from 0.32 to 0.54 on Open-Plan, 0.25 to
0.46 on Beam House and 0.37 to 0.54 on House. Annotation hints: `"shape": "square"` draws a boxy
armchair (tufted) instead of a round tub chair, `"shape": "plain"` an island without cooktop and
hood. In a photoreal scene, **Overlay** (next to Photoreal / Blueprint) draws the blueprint's lines
over the splats to check the two line up.

Besides furniture, the blueprints carry the smaller items in each room (Open-Plan 59 pieces,
House 42, Beam House 35): plants, floor and table lamps, pendant lights, chandeliers, vases,
bowls and decor, sculptures, cushions, rugs, wall art, mirrors, curtains, sideboards, fireplaces,
an ottoman, a piano, the refrigerator and shelving. Each is marked the same way and drawn as a
small line shape from where it sits: on the floor, on a surface, on a wall or hanging from the
ceiling. Thin or see-through items show mostly what is behind them in the depth map, so an item's
distance is tied to geometry already measured (furniture first, then items): the sofa or table
its centre line passes over (cushions, vases, table lamps), the table a light hangs over, the
floor where its box's bottom edge meets it (only when that line points clearly down), or the
blueprint wall it hangs on; an item far beyond the furniture its line crosses stands just behind
it. The depth map is the fallback (and decides for wall items when the blueprint wall disagrees
by over 1 m). An item marked in several frames is measured from the nearest view only. Vases,
bowls, lamps, plants and cushions are then set down on the surface they stand on. `--measure --only-new` measures only annotation ids that `furniture.json` does
not have yet, so items can be added without moving pieces that are already right. When the
long pieces (sofas, tables, islands) agree on a wall direction more than 10° from the one the
points give, the blueprint's grid follows them (Open-Plan: its white walls and glass left the
wall estimate nearly flat).

Every build also cuts each piece's photo from the frame where its box is largest
(`artifacts/objects/<id>.jpg`, plus `views.json` with that frame's capture spot) when the
source keyframes are present, and `blueprint.json` gives each piece a name, dimensions and which
of them were measured; each furniture line carries its piece's index, so the viewer can pick it.
A measured height more than 35% off the type's typical height (the box also caught a wall or
lamp behind it) is reported as typical.

## Capture quality report
`artifacts/scene_report.json` records registered images, points, mean reprojection error and
`largest_view_direction_gap_deg`: the widest horizontal angle no camera looked towards. Under
about 45° means you can turn around anywhere in the scene and see captured content. To get there,
pan slowly through a full turn in each room while walking.

## Sample data
`tests/data/library_sample/` holds 30 frames of a public indoor video (CC BY-SA 4.0, see ATTRIBUTION.txt).

## Known limits
- Scale is assumed (camera height) until calibrated. Calibration stores a factor (`artifacts/scale_calibration.json`) applied to measurements only; it does not rescale the geometry.
- Sparse SfM point cloud only; train 3DGS (e.g. gsplat, Apache-2.0) on a GPU for photoreal output.
- POI labels are supplied in `poi_database.json`; CLIP embedding fusion exists in `semantics/` but is not wired to an auto-labeller.
