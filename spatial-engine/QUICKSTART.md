# Spatial Engine - Quick Start

## One-Command Launch

Run the entire system (verification tests + backend + frontend) with a single command:

```bash
cd spatial-engine
python launch.py
```

This will:

1. ✓ Run all 4 end-to-end verification tests
2. 🚀 Start the FastAPI backend server (http://localhost:8000)
3. 🎨 Start the Vite frontend dev server (http://localhost:5173)

## Command Options

```bash
# Run tests only, don't start servers
python launch.py --test-only

# Start servers without running tests
python launch.py --no-test

# Full run (default)
python launch.py
```

## Access Points

Once running:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs

## Stop All Services

Press **Ctrl+C** to gracefully stop all services.

---

## Manual Commands (if needed)

### Run Tests Only
```bash
python -m pytest tests/ -v
# or
python tests/test_e2e_verification.py
```

### Start Backend Only
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend Only
```bash
cd frontend
npm install
npm run dev
```

### Docker (if Dockerfile exists)
```bash
docker-compose up
```

---

## Verification Results

Last run:
```
✓ Gravity Aligner — RANSAC floor alignment (scale=2.6364)
✓ Semantic POI Back-Projection — 2D→3D voxel fusion (1 cluster)
✓ Pathplanner (Synthetic Room) — 318 waypoints, 12.18m, 0.403m clearance
✓ Snap to Nearest Walkable — obstacle-to-free-space snapping
```

---

## Architecture

- **Backend**: FastAPI + Pydantic + NumPy/SciPy/Trimesh
- **Frontend**: Three.js + gaussian-splats-3d + three-mesh-bvh
- **Pipeline**: COLMAP → Gravity Align → 3DGS Train → TSDF Mesh → ESDF Navgrid
- **Features**: Semantic POI search, collision-free pathplanning, interactive 3D viewer
