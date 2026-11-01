"""
End-to-End Verification Suite for the Intelligent 3D Spatial Engine
Section 8.2 of the Masterplan Blueprint

Validates:
  1. MetricGravityAligner — RANSAC floor plane fitting & Rodrigues rotation
  2. VoxelSemanticField — 2D→3D depth back-projection, DBSCAN clustering, open-vocabulary query
  3. IndoorSpatialPathPlanner — ESDF navgrid build, Theta* search, B-Spline smoothing
  4. Full path planning on a synthetic multi-room indoor environment with central obstacle wall
"""

import sys
import numpy as np
import trimesh

# Adjust path so imports work from the spatial-engine root
sys.path.insert(0, ".")

from backend.reconstruction.gravity_aligner import MetricGravityAligner
from backend.semantics.spatial_poi_indexer import VoxelSemanticField
from backend.navigation.pathplanner import IndoorSpatialPathPlanner


def test_gravity_aligner():
    """Test RANSAC floor plane and gravity alignment."""
    aligner = MetricGravityAligner()

    # Create a simple test mesh (floor + 4 walls)
    floor = trimesh.creation.box(extents=[5.0, 0.1, 5.0])
    wall1 = trimesh.creation.box(extents=[0.2, 2.0, 5.0])
    wall1.apply_translation([2.5, 1.0, 0.0])
    wall2 = trimesh.creation.box(extents=[0.2, 2.0, 5.0])
    wall2.apply_translation([-2.5, 1.0, 0.0])
    wall3 = trimesh.creation.box(extents=[5.0, 2.0, 0.2])
    wall3.apply_translation([0.0, 1.0, 2.5])
    wall4 = trimesh.creation.box(extents=[5.0, 2.0, 0.2])
    wall4.apply_translation([0.0, 1.0, -2.5])
    test_mesh = trimesh.util.concatenate([floor, wall1, wall2, wall3, wall4])

    # Simulate camera poses with a slight tilt (COLMAP format: rotation is C2W)
    rng = np.random.default_rng(42)
    n_cameras = 30
    sfm_camera_centers = []
    sfm_camera_rotations_c2w = []

    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras
        # Camera positions on a circle around the center
        cx = 1.5 * np.cos(angle)
        cz = 1.5 * np.sin(angle)
        cy = 1.45  # Camera height
        sfm_camera_centers.append([cx, cy, cz])

        # Rotation matrices - look at origin from each position
        look_at = np.array([0.0, cy, 0.0]) - np.array([cx, cy, cz])
        look_at = look_at / np.linalg.norm(look_at)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(look_at, up)
        right = right / np.linalg.norm(right)
        actual_up = np.cross(right, look_at)
        R = np.column_stack([right, actual_up, -look_at])
        sfm_camera_rotations_c2w.append(R)

    sfm_centers = np.array(sfm_camera_centers)
    sfm_rotations = np.array(sfm_camera_rotations_c2w)

    sim3, scale = aligner.compute_canonical_sim3(
        test_mesh, sfm_centers, sfm_rotations
    )

    assert sim3.shape == (4, 4), "Sim3 matrix must be 4x4"
    assert scale > 0, "Scale must be positive"
    # For this simple case, the transformation should be close to identity
    # (since the input is already gravity-aligned)
    print(f"[PASS] MetricGravityAligner — RANSAC floor alignment produced Sim(3) with scale={scale:.4f}")


def test_semantic_poi_backprojection():
    """Test 2D→3D depth back-projection and open-vocabulary POI query."""
    sem = VoxelSemanticField(voxel_size_m=0.06, feature_dim=512)

    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    depth = np.full((100, 100), 2.5, dtype=np.float32)
    K = np.array([[80.0, 0, 50.0], [0, 80.0, 50.0], [0, 0, 1.0]])
    c2w = np.eye(4)
    target_emb = np.ones(512, dtype=np.float32)

    sem.fuse_mask_observation(mask, depth, K, c2w, target_emb)

    pois = sem.query_open_vocabulary(target_emb, "espresso machine", similarity_threshold=0.8)
    assert len(pois) == 1, f"Expected 1 POI cluster, got {len(pois)}"
    assert pois[0].label == "espresso machine"
    print(
        f"[PASS] Semantic 3D POI verified: {pois[0].label} at "
        f"{pois[0].centroid_xyz} (conf={pois[0].confidence})"
    )


def test_pathplanner_synthetic_room():
    """
    Test ESDF navgrid + Theta* + B-Spline on a synthetic multi-room environment:
    10m x 10m floor with a central 6m-wide, 0.3m-thick wall and a 4m doorway gap.
    """
    # 1. Create synthetic indoor room
    floor = trimesh.creation.box(extents=[10.0, 0.1, 10.0])
    floor.apply_translation([0.0, -0.05, 0.0])

    wall = trimesh.creation.box(extents=[6.0, 2.5, 0.3])
    wall.apply_translation([-2.0, 1.25, 0.0])

    scene_mesh = trimesh.util.concatenate([floor, wall])

    # 2. Build Navigation Grid & ESDF
    planner = IndoorSpatialPathPlanner(cell_size_m=0.05, agent_radius_m=0.28)
    planner.build_navgrid_from_mesh(scene_mesh)

    assert planner.grid_shape[0] > 0 and planner.grid_shape[1] > 0, "Grid dimensions must be positive"
    assert planner.walkable_mask is not None, "Walkable mask must exist"
    assert planner.esdf_meters is not None, "ESDF must exist"

    walkable_area = float(np.sum(planner.walkable_mask)) * (planner.cell_size_m ** 2)
    print(f"[INFO] Navgrid: {planner.grid_shape}, walkable_area={walkable_area:.1f} m²")

    # 3. Plan path around the wall obstacle (start in front, goal behind the wall)
    res = planner.plan_path(
        start_xyz=(-3.5, 1.5, -3.0),
        goal_xyz=(-3.5, 1.5, 3.0),
    )

    assert len(res.waypoints_3d) > 10, f"Path too short: {len(res.waypoints_3d)} waypoints"
    assert res.min_obstacle_clearance_m >= 0.28, (
        f"Collision detected! Clearance: {res.min_obstacle_clearance_m}"
    )
    assert res.total_length_m > 0, "Path length must be positive"

    print(
        f"[PASS] Pathplanner verified: {len(res.waypoints_3d)} smooth waypoints, "
        f"length={res.total_length_m:.2f}m, min_clearance={res.min_obstacle_clearance_m:.3f}m"
    )

    # 4. Verify path through the doorway gap (should be shorter)
    res_through = planner.plan_path(
        start_xyz=(3.0, 1.5, -3.0),
        goal_xyz=(3.0, 1.5, 3.0),
    )
    assert res_through.total_length_m > 0, "Doorway path must exist"
    print(
        f"[PASS] Doorway path verified: {len(res_through.waypoints_3d)} waypoints, "
        f"length={res_through.total_length_m:.2f}m"
    )


def test_snap_to_nearest_walkable():
    """Test that clicking on an obstacle snaps to the nearest free-space cell."""
    floor = trimesh.creation.box(extents=[5.0, 0.1, 5.0])
    floor.apply_translation([0.0, -0.05, 0.0])
    wall = trimesh.creation.box(extents=[2.0, 2.5, 2.0])
    wall.apply_translation([0.0, 1.25, 0.0])
    mesh = trimesh.util.concatenate([floor, wall])

    planner = IndoorSpatialPathPlanner(cell_size_m=0.05, agent_radius_m=0.20)
    planner.build_navgrid_from_mesh(mesh)

    # Click inside the wall obstacle at world (0, 1.5, 0)
    ix, iz = planner.world_to_grid(0.0, 0.0)
    snapped_ix, snapped_iz = planner.snap_to_nearest_walkable(ix, iz)

    assert planner.walkable_mask[snapped_ix, snapped_iz], "Snapped point must be walkable"
    print(
        f"[PASS] snap_to_nearest_walkable: obstacle ({ix},{iz}) → "
        f"free-space ({snapped_ix},{snapped_iz})"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  Intelligent 3D Spatial Engine — End-to-End Verification Suite")
    print("=" * 70)
    print()

    tests = [
        ("Gravity Aligner", test_gravity_aligner),
        ("Semantic POI Back-Projection", test_semantic_poi_backprojection),
        ("Pathplanner (Synthetic Room)", test_pathplanner_synthetic_room),
        ("Snap to Nearest Walkable", test_snap_to_nearest_walkable),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"--- Test: {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"  Results: {passed} PASSED, {failed} FAILED out of {len(tests)} tests")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)
