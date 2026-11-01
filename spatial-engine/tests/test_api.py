"""API tests against the generated demo scene (run create_demo_scene.py first)."""
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, "create_demo_scene.py"], cwd=ROOT, check=True)
    from backend.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_initialize_and_path(client):
    r = client.post("/api/v1/scenes/demo_office/initialize")
    assert r.status_code == 200 and r.json()["walkable_area_m2"] > 20
    p = client.post("/api/v1/navigation/plan-path", json={
        "scene_id": "demo_office", "start_xyz": [1, 1.5, 1], "goal_xyz": [8, 0, 6]})
    assert p.status_code == 200
    assert p.json()["min_obstacle_clearance_m"] >= 0.28


def test_poi_search(client):
    client.post("/api/v1/scenes/demo_office/initialize")
    r = client.post("/api/v1/semantics/search", json={"scene_id": "demo_office", "query_text": "espresso machine"})
    assert r.json()[0]["label"] == "espresso machine"
    assert client.post("/api/v1/semantics/search", json={
        "scene_id": "demo_office", "query_text": "zebra"}).json() == []


def test_unknown_scene(client):
    assert client.post("/api/v1/scenes/nope/initialize").status_code == 404


def test_map_status_and_capture(client):
    import base64
    import numpy as np
    client.post("/api/v1/scenes/demo_office/initialize")
    m = client.get("/api/v1/scenes/demo_office/map").json()
    status = np.frombuffer(base64.b64decode(m["status_b64"]), dtype=np.uint8)
    assert status.size == m["nx"] * m["nz"]
    assert set(np.unique(status)) <= {0, 1, 2}
    assert (status == 1).sum() > 0 and (status == 2).sum() > 0


def test_map_requires_init(client):
    assert client.get("/api/v1/scenes/not_loaded/map").status_code == 400


def test_collision_mesh_keeps_walked_path_free():
    """Obstacle points on the walked camera path are SfM noise and must not block navigation."""
    import numpy as np
    from backend.navigation.pathplanner import IndoorSpatialPathPlanner
    from backend.reconstruction.pycolmap_pipeline import build_collision_mesh

    path = np.array([[x, 1.45, 0.0] for x in np.linspace(0, 6, 13)])
    noise_on_path = np.array([[3.0, 1.0, 0.1]])               # within 0.45 m of the path
    real_wall = np.array([[x, h, 2.0] for x in np.linspace(0, 6, 30) for h in (0.5, 1.0, 1.5)])
    mesh = build_collision_mesh(np.vstack([noise_on_path, real_wall]), path)

    planner = IndoorSpatialPathPlanner()
    planner.build_navgrid_from_mesh(mesh)
    assert all(planner.status_grid[planner.world_to_grid(p[0], p[2])] == 1 for p in path)
    assert planner.status_grid[planner.world_to_grid(3.0, 2.0)] == 2   # the wall still blocks


def test_scene_list_for_switcher(client):
    scenes = {s["id"]: s for s in client.get("/api/v1/scenes").json()["scenes"]}
    assert "demo_office" not in scenes  # hidden test fixture; its API still works (tests above)
    assert "library" not in scenes
    for sc in scenes.values():  # the viewer loads straight from these flags, without probing
        assert set(sc["files"]) >= {"ksplat", "ply", "points", "mesh", "blueprint"}
    if "house_1" in scenes:  # shipped blueprint duplicate of the house capture
        assert scenes["house_1"]["style"] == "blueprint"
        # twins: the switcher lists the house once and the toggle flips between the two
        assert scenes["house_1"]["place"] == "house"
        assert scenes["house"]["twins"] == {"blueprint": "house_1"}
        assert scenes["house_1"]["twins"] == {"photoreal": "house"}
    if "open_plan" in scenes:  # scene.json picks the capture spot the viewer opens on
        assert scenes["open_plan"]["start_view"] == 4
        assert scenes["open_plan_1"]["twins"] == {"photoreal": "open_plan"}


def test_scene_files_are_cacheable_and_isolated(client):
    """Artifacts revalidate by ETag (304, no body) and every response carries the headers that
    let the splat sorter share memory with its worker."""
    r = client.get("/scenes/demo_office/artifacts/poi_database.json")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cross-origin-embedder-policy"] == "credentialless"
    again = client.get("/scenes/demo_office/artifacts/poi_database.json", headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304
    bp = client.get("/scenes/house_1/artifacts/blueprint.json")  # large JSON is gzipped
    if bp.status_code == 200:
        assert bp.headers.get("content-encoding") == "gzip"
    page = client.get("/")  # the built page revalidates, so a rebuild's new asset names load
    if page.status_code == 200 and page.headers["content-type"].startswith("text/html"):
        assert page.headers["cache-control"] == "no-cache"


def test_initialize_reuses_navgrid(client):
    first = client.post("/api/v1/scenes/demo_office/initialize").json()
    from backend.main import LOADED_PLANNERS
    planner = LOADED_PLANNERS["demo_office"]
    assert client.post("/api/v1/scenes/demo_office/initialize").json() == first
    assert LOADED_PLANNERS["demo_office"] is planner  # not rebuilt for the same mesh


def test_blueprint_lines_are_drafted():
    """The house blueprint is lines only; its walls are snapped to the dominant wall direction
    and its furniture is drawn as recognisable pieces."""
    import json
    import math
    bp_file = ROOT / "data" / "scenes" / "house_1" / "artifacts" / "blueprint.json"
    if not bp_file.exists():
        pytest.skip("house_1 blueprint not built")
    bp = json.loads(bp_file.read_text())
    assert bp["counts"]["outline_corners"] >= 4
    if (bp_file.parent / "furniture.json").exists():
        assert {"sectional", "pool_table", "kitchen_island"} <= {o["type"] for o in bp["objects"]}
    c, s = math.cos(bp["yaw_rad"]), math.sin(bp["yaw_rad"])
    for seg in bp["segments"]:
        x1, y1, z1, x2, y2, z2, kind = seg[:7]
        assert kind in {"outline", "corner", "wall", "object"}
        if kind == "object":  # furniture drawings have round parts; each line names its piece
            assert 0 <= seg[7] < len(bp["objects"])
            continue
        # in the plan frame every edge runs along one axis: horizontal, or straight up
        u = (x2 - x1) * c - (z2 - z1) * s
        v = (x2 - x1) * s + (z2 - z1) * c
        flat = abs(y2 - y1) < 1e-3
        assert (flat and min(abs(u), abs(v)) < 0.02) or (abs(u) < 0.02 and abs(v) < 0.02)


def test_blueprint_objects_are_described(client):
    """Every blueprint piece carries what the selection card shows: a name, its dimensions in
    metres, which of them were measured, and a photo cropped from the capture when one exists."""
    import json
    art = ROOT / "data" / "scenes" / "house_1" / "artifacts"
    if not (art / "blueprint.json").exists() or not (art / "furniture.json").exists():
        pytest.skip("house_1 blueprint not built")
    bp = json.loads((art / "blueprint.json").read_text())
    for o in bp["objects"]:
        assert o["name"] and o["dims"]["width"] > 0
        assert set(o["measured"]) <= set(o["dims"])
        if "photo" in o:
            assert (art / o["photo"]).exists()
            r = client.get(f"/scenes/house_1/artifacts/{o['photo']}")
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
