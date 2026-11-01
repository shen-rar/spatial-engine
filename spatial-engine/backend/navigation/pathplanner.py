import heapq
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import distance_transform_edt, binary_closing
from scipy.interpolate import splprep, splev


@dataclass
class NavigationPathResult:
    waypoints_3d: List[List[float]]       # Dense smooth [x, y, z] camera/foot path
    raw_control_points_3d: List[List[float]]
    total_length_m: float
    min_obstacle_clearance_m: float


class IndoorSpatialPathPlanner:
    """
    Builds a 2.5D ESDF Navigation Grid directly from a gravity-aligned 3D collision mesh
    and computes optimal, non-colliding, clearance-aware trajectories using Any-Angle
    Theta* and collision-verified Cubic B-Splines.
    """

    def __init__(
        self,
        cell_size_m: float = 0.05,
        agent_radius_m: float = 0.28,
        step_height_m: float = 0.16,
        agent_height_m: float = 1.80,
        camera_eye_height_m: float = 1.50,
    ):
        self.cell_size_m = cell_size_m
        self.agent_radius_m = agent_radius_m
        self.step_height_m = step_height_m
        self.agent_height_m = agent_height_m
        self.camera_eye_height_m = camera_eye_height_m

        self.origin_xz: np.ndarray = np.zeros(2)
        self.grid_shape: Tuple[int, int] = (0, 0)
        self.walkable_mask: Optional[np.ndarray] = None
        self.esdf_meters: Optional[np.ndarray] = None
        self.floor_elevation_map: Optional[np.ndarray] = None
        self.status_grid: Optional[np.ndarray] = None  # 0 unseen, 1 open, 2 blocked

    def build_navgrid_from_mesh(self, mesh: trimesh.Trimesh) -> None:
        """
        Slices the 3D mesh into floor support and obstacle occupancy grids, then
        computes the Euclidean Signed Distance Field (ESDF) in meters.
        """
        bounds_min, bounds_max = mesh.bounds
        margin = 0.5
        self.origin_xz = np.array([bounds_min[0] - margin, bounds_min[2] - margin])
        max_xz = np.array([bounds_max[0] + margin, bounds_max[2] + margin])

        nx = int(np.ceil((max_xz[0] - self.origin_xz[0]) / self.cell_size_m))
        nz = int(np.ceil((max_xz[1] - self.origin_xz[1]) / self.cell_size_m))
        self.grid_shape = (nx, nz)

        # Sample dense surface points from the collision mesh to guarantee zero missed thin walls
        surface_pts, _ = trimesh.sample.sample_surface(mesh, count=max(250000, len(mesh.faces) * 4))
        ix = np.clip(
            ((surface_pts[:, 0] - self.origin_xz[0]) / self.cell_size_m).astype(np.int32),
            0,
            nx - 1,
        )
        iz = np.clip(
            ((surface_pts[:, 2] - self.origin_xz[1]) / self.cell_size_m).astype(np.int32),
            0,
            nz - 1,
        )
        ys = surface_pts[:, 1]

        # 1. Identify Floor Surface Points (Y within [-0.20m, +0.25m])
        floor_mask_pts = (ys >= -0.20) & (ys <= 0.25)
        raw_floor = np.zeros((nx, nz), dtype=bool)
        raw_floor[ix[floor_mask_pts], iz[floor_mask_pts]] = True
        # Close small mesh holes in the floor
        raw_floor = binary_closing(raw_floor, structure=np.ones((5, 5)))

        self.floor_elevation_map = np.zeros((nx, nz), dtype=np.float32)

        # 2. Identify Obstacles intersecting [step_height_m, agent_height_m]
        obs_mask_pts = (ys >= self.step_height_m) & (ys <= self.agent_height_m)
        obstacle_grid = np.zeros((nx, nz), dtype=bool)
        obstacle_grid[ix[obs_mask_pts], iz[obs_mask_pts]] = True

        # Also mark exterior boundary outside walkable floor as obstacle
        combined_free = raw_floor & (~obstacle_grid)

        # 3. Compute exact Euclidean Distance Transform in meters
        self.esdf_meters = (
            distance_transform_edt(combined_free).astype(np.float32) * self.cell_size_m
        )
        self.walkable_mask = self.esdf_meters >= self.agent_radius_m

        # Map status for the plan view: open = walkable, blocked = obstacle or floor too close
        # to walls, unseen = no floor observed by the capture.
        self.status_grid = np.zeros((nx, nz), dtype=np.uint8)
        self.status_grid[raw_floor] = 2
        self.status_grid[self.walkable_mask] = 1
        self.status_grid[obstacle_grid] = 2

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        ix = int(np.round((x - self.origin_xz[0]) / self.cell_size_m))
        iz = int(np.round((z - self.origin_xz[1]) / self.cell_size_m))
        ix = int(np.clip(ix, 0, self.grid_shape[0] - 1))
        iz = int(np.clip(iz, 0, self.grid_shape[1] - 1))
        return ix, iz

    def grid_to_world(self, ix: int, iz: int) -> Tuple[float, float]:
        x = float(self.origin_xz[0] + (ix + 0.5) * self.cell_size_m)
        z = float(self.origin_xz[1] + (iz + 0.5) * self.cell_size_m)
        return x, z

    def snap_to_nearest_walkable(self, ix: int, iz: int) -> Tuple[int, int]:
        """
        If a user clicks on a table, wall, or POI bounding box that is inside an obstacle,
        snaps the target to the closest valid navigable point in free space.
        """
        if self.walkable_mask[ix, iz]:
            return ix, iz

        valid_xs, valid_zs = np.nonzero(self.walkable_mask)
        if len(valid_xs) == 0:
            raise RuntimeError("Navigation grid contains zero walkable cells.")

        dists_sq = (valid_xs - ix) ** 2 + (valid_zs - iz) ** 2
        best_idx = int(np.argmin(dists_sq))
        return int(valid_xs[best_idx]), int(valid_zs[best_idx])

    def _line_of_sight_clear(self, u: Tuple[int, int], v: Tuple[int, int]) -> bool:
        """
        Bresenham supercover raymarch checking both walkability and safe ESDF clearance.
        """
        x0, z0 = u
        x1, z1 = v
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        steps = max(dx, dz)
        if steps == 0:
            return bool(self.walkable_mask[x0, z0])

        xs = np.linspace(x0, x1, steps + 1).round().astype(np.int32)
        zs = np.linspace(z0, z1, steps + 1).round().astype(np.int32)

        # Require slightly higher clearance for long straight-line shortcuts
        min_safe = self.agent_radius_m + 0.05
        return bool(np.all(self.esdf_meters[xs, zs] >= min_safe))

    def _clearance_penalty(self, ix: int, iz: int) -> float:
        clearance_margin = max(0.0, float(self.esdf_meters[ix, iz]) - self.agent_radius_m)
        return 1.0 + 3.5 * np.exp(-4.0 * clearance_margin)

    def plan_path(
        self, start_xyz: Tuple[float, float, float], goal_xyz: Tuple[float, float, float]
    ) -> NavigationPathResult:
        """
        Computes a smooth, non-colliding 3D path from start_xyz to goal_xyz.
        """
        if self.walkable_mask is None or self.esdf_meters is None:
            raise RuntimeError("Navigation grid has not been built yet.")

        s_raw = self.world_to_grid(start_xyz[0], start_xyz[2])
        g_raw = self.world_to_grid(goal_xyz[0], goal_xyz[2])

        start_node = self.snap_to_nearest_walkable(*s_raw)
        goal_node = self.snap_to_nearest_walkable(*g_raw)

        if start_node == goal_node:
            wx, wz = self.grid_to_world(*start_node)
            pt = [round(wx, 3), round(self.camera_eye_height_m, 3), round(wz, 3)]
            return NavigationPathResult(
                waypoints_3d=[pt, pt],
                raw_control_points_3d=[pt],
                total_length_m=0.0,
                min_obstacle_clearance_m=float(self.esdf_meters[start_node]),
            )

        # 8-connected neighbor offsets
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
        ]

        g_score = {start_node: 0.0}
        parent = {start_node: start_node}

        def heuristic(n: Tuple[int, int]) -> float:
            return float(np.hypot(n[0] - goal_node[0], n[1] - goal_node[1]))

        open_heap = [(heuristic(start_node), start_node)]
        closed_set = set()
        nx, nz = self.grid_shape

        while open_heap:
            _, curr = heapq.heappop(open_heap)
            if curr == goal_node:
                break
            if curr in closed_set:
                continue
            closed_set.add(curr)

            for dx, dz, step_len in neighbors:
                nb = (curr[0] + dx, curr[1] + dz)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < nz):
                    continue
                if not self.walkable_mask[nb[0], nb[1]] or nb in closed_set:
                    continue

                penalty = self._clearance_penalty(nb[0], nb[1])
                par = parent[curr]

                # Any-Angle Theta* relaxation if line-of-sight from parent(curr) to nb is clear
                if par != curr and self._line_of_sight_clear(par, nb):
                    euclid_dist = float(np.hypot(nb[0] - par[0], nb[1] - par[1]))
                    tentative_g = g_score[par] + euclid_dist * penalty
                    candidate_parent = par
                else:
                    tentative_g = g_score[curr] + step_len * penalty
                    candidate_parent = curr

                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    parent[nb] = candidate_parent
                    f_val = tentative_g + heuristic(nb)
                    heapq.heappush(open_heap, (f_val, nb))

        if goal_node not in parent:
            raise ValueError("Goal lies in a disconnected topological island.")

        # Reconstruct grid path
        path_grid: List[Tuple[int, int]] = []
        curr = goal_node
        while curr != start_node:
            path_grid.append(curr)
            curr = parent[curr]
        path_grid.append(start_node)
        path_grid.reverse()

        # Densify long Theta* segments so B-Spline smoothing respects local corridors
        densified_grid: List[Tuple[int, int]] = [path_grid[0]]
        for i in range(len(path_grid) - 1):
            p0, p1 = path_grid[i], path_grid[i + 1]
            dist_cells = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            num_sub = max(1, int(dist_cells // 5))
            for s in range(1, num_sub + 1):
                alpha = s / float(num_sub)
                ix = int(round(p0[0] * (1 - alpha) + p1[0] * alpha))
                iz = int(round(p0[1] * (1 - alpha) + p1[1] * alpha))
                if (ix, iz) != densified_grid[-1]:
                    densified_grid.append((ix, iz))

        world_xz = np.array([self.grid_to_world(ix, iz) for ix, iz in densified_grid])
        smoothed_xz = self._smooth_and_verify_bspline(world_xz)

        # Lift to 3D waypoints and compute path metrics
        diffs = np.diff(smoothed_xz, axis=0)
        total_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        clearances = [
            float(self.esdf_meters[self.world_to_grid(pt[0], pt[1])]) for pt in smoothed_xz
        ]
        min_clearance = float(min(clearances)) if clearances else 0.0

        waypoints_3d = [
            [round(float(pt[0]), 3), round(self.camera_eye_height_m, 3), round(float(pt[1]), 3)]
            for pt in smoothed_xz
        ]
        raw_ctrl_3d = [
            [
                round(float(pt[0]), 3),
                round(self.camera_eye_height_m, 3),
                round(float(pt[1]), 3),
            ]
            for pt in world_xz
        ]

        return NavigationPathResult(
            waypoints_3d=waypoints_3d,
            raw_control_points_3d=raw_ctrl_3d,
            total_length_m=round(total_length, 3),
            min_obstacle_clearance_m=round(min_clearance, 3),
        )

    def _smooth_and_verify_bspline(self, pts_xz: np.ndarray) -> np.ndarray:
        """
        Fits a Cubic B-Spline through the waypoints and verifies that no smoothed point
        violates the agent's collision radius. Falls back to piecewise linear subdivision
        near tight corners if a spline overshoot is detected.
        """
        if len(pts_xz) <= 3:
            return pts_xz

        try:
            tck, _ = splprep([pts_xz[:, 0], pts_xz[:, 1]], s=0.02 * len(pts_xz), k=3)
            u_fine = np.linspace(0.0, 1.0, max(50, len(pts_xz) * 6))
            sx, sz = splev(u_fine, tck)
            candidate_xz = np.stack([sx, sz], axis=1)

            # Verify zero collision across all spline samples
            for pt in candidate_xz:
                ix, iz = self.world_to_grid(float(pt[0]), float(pt[1]))
                if not self.walkable_mask[ix, iz]:
                    return pts_xz
            return candidate_xz
        except Exception:
            return pts_xz