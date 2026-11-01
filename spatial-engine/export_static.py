"""Export the viewer and its scenes as a static site (Cloudflare Pages, Netlify, any host
that reads `_headers` / `_redirects`).

    cd frontend && npm run build && cd ..
    python export_static.py site/
    npx wrangler pages deploy site --project-name spatial-engine

The backend's read-only answers are written as files and mapped onto the API paths with
`_redirects` rewrites; routes are planned in the browser on the exported floor-plan grid.
Only the files the viewer loads are copied (no training .ply files, keyframes or COLMAP).
"""
import argparse
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import DATA_ROOT, FRONTEND_DIST, app

SCENE_FILES = ("visual_splats.ksplat", "collision_mesh.glb", "cameras.json", "reference.jpg",
               "blueprint.json", "furniture.json")
MAX_FILE = 25 * 1024 * 1024  # Cloudflare Pages per-file limit

HEADERS = """/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: credentialless
  Cross-Origin-Resource-Policy: cross-origin
/assets/*
  Cache-Control: public, max-age=31536000, immutable
/scenes/*
  Cache-Control: public, max-age=3600
"""

REDIRECTS = """/api/v1/scenes /api/v1/scenes.json 200
/api/v1/scenes/:id/map /api/v1/scenes/:id/map.json 200
/api/v1/scenes/:id/calibration /api/v1/scenes/:id/calibration.json 200
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    if not (FRONTEND_DIST / "index.html").exists():
        raise SystemExit("frontend/dist is missing: run `npm run build` in frontend/ first")
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(FRONTEND_DIST, out)

    client = TestClient(app)
    scenes = client.get("/api/v1/scenes").json()
    api = out / "api" / "v1" / "scenes"
    for sc in scenes["scenes"]:
        sid = sc["id"]
        src, dst = DATA_ROOT / sid, out / "scenes" / sid
        (dst / "artifacts").mkdir(parents=True)
        if (src / "scene.json").exists():
            shutil.copy2(src / "scene.json", dst / "scene.json")
        for f in SCENE_FILES:
            p = src / "artifacts" / f
            if p.exists():
                if p.stat().st_size > MAX_FILE:
                    raise SystemExit(f"{p} is over the 25 MB per-file limit")
                shutil.copy2(p, dst / "artifacts" / f)
        if (src / "artifacts" / "objects").is_dir():
            shutil.copytree(src / "artifacts" / "objects", dst / "artifacts" / "objects")
        # the viewer picks its layers from these flags: say only what was copied
        sc["files"].update(ply=False, points=False)

        (api / sid).mkdir(parents=True)
        client.post(f"/api/v1/scenes/{sid}/initialize").raise_for_status()
        for name in ("map", "calibration"):
            r = client.get(f"/api/v1/scenes/{sid}/{name}")
            r.raise_for_status()
            (api / sid / f"{name}.json").write_text(r.text, encoding="utf-8")
        print(f"  {sid}: {sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()) / 1e6:.1f} MB")

    (out / "api" / "v1" / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    (out / "_headers").write_text(HEADERS, encoding="utf-8")
    (out / "_redirects").write_text(REDIRECTS, encoding="utf-8")
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"{out}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
