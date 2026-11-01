import numpy as np
import trimesh
from typing import Tuple, List, Optional


class MetricGravityAligner:
    """
    Computes a rigid Sim(3) transformation (scale, rotation, translation) that aligns
    an arbitrary SfM/Mesh/3DGS reconstruction so that:
      1. 1.0 coordinate unit == 1.0 physical meter.
      2. The floor plane lies strictly on Y = 0 with surface normal pointing to +Y [0, 1, 0].
      3. The scene XY/XZ centroid is centered at (0, 0, 0) for numerical stability in WebGL.
    """

    def __init__(self, default_camera_height_m: float = 1.45):
        self.default_camera_height_m = default_camera_height_m

    @staticmethod
    def umeyama_sim3(
        src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solves min_{s, R, t} || dst - (s * R @ src + t) ||^2
        src_pts, dst_pts: (N, 3) numpy arrays.
        """
        assert src_pts.shape == dst_pts.shape and src_pts.shape[0] >= 3
        n = src_pts.shape[0]
        mu_src = np.mean(src_pts, axis=0)
        mu_dst = np.mean(dst_pts, axis=0)

        src_centered = src_pts - mu_src
        dst_centered = dst_pts - mu_dst

        var_src = np.sum(src_centered ** 2) / n
        cov = (dst_centered.T @ src_centered) / n

        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0

        R = U @ S @ Vt
        scale = float(np.trace(np.diag(D) @ S) / var_src)
        t = mu_dst - scale * (R @ mu_src)
        return scale, R, t

    @staticmethod
    def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Finds the 3x3 rotation matrix that aligns vec1 to vec2."""
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))

        if s < 1e-8:
            return np.eye(3) if c > 0 else -np.eye(3)

        kmat = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])
        return np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s ** 2))

    def fit_ransac_floor_plane(
        self,
        vertices: np.ndarray,
        camera_centers: np.ndarray,
        camera_up_hint: np.ndarray,
        iterations: int = 1000,
        distance_threshold: float = 0.04,
    ) -> Tuple[np.ndarray, float]:
        """
        Fits the dominant floor plane using RANSAC constrained by the average camera up vector.
        Returns (unit_normal, plane_offset_d) where n . x + d = 0.
        """
        up_unit = camera_up_hint / np.linalg.norm(camera_up_hint)
        cam_heights = camera_centers @ up_unit
        median_cam_h = np.median(cam_heights)

        # Filter vertices that lie below the cameras along the approximate up direction
        vert_heights = vertices @ up_unit
        floor_candidates = vertices[vert_heights < median_cam_h]
        if len(floor_candidates) < 100:
            floor_candidates = vertices

        # Keep lowest 35% of candidate points
        cutoff = np.percentile(floor_candidates @ up_unit, 35.0)
        low_pts = floor_candidates[(floor_candidates @ up_unit) <= cutoff]
        if len(low_pts) < 50:
            low_pts = floor_candidates

        best_inliers = -1
        best_normal = up_unit
        best_d = 0.0
        rng = np.random.default_rng(42)

        for _ in range(iterations):
            idx = rng.choice(len(low_pts), size=3, replace=False)
            p1, p2, p3 = low_pts[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal /= norm_len

            # Orient normal to match camera up hint
            if np.dot(normal, up_unit) < 0:
                normal = -normal

            # Reject planes tilted more than 25 degrees from camera up hint
            if np.dot(normal, up_unit) < np.cos(np.deg2rad(25.0)):
                continue

            d = -float(np.dot(normal, p1))
            dists = np.abs(low_pts @ normal + d)
            inliers = int(np.sum(dists < distance_threshold))

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_d = d

        # Refine plane using SVD on all inliers
        inlier_mask = np.abs(low_pts @ best_normal + best_d) < distance_threshold
        if np.sum(inlier_mask) >= 10:
            inlier_pts = low_pts[inlier_mask]
            centroid = np.mean(inlier_pts, axis=0)
            _, _, Vt = np.linalg.svd(inlier_pts - centroid)
            refined_normal = Vt[-1]
            if np.dot(refined_normal, up_unit) < 0:
                refined_normal = -refined_normal
            best_normal = refined_normal / np.linalg.norm(refined_normal)
            best_d = -float(np.dot(best_normal, centroid))

        return best_normal, best_d

    def compute_canonical_sim3(
        self,
        mesh: trimesh.Trimesh,
        sfm_camera_centers: np.ndarray,
        sfm_camera_rotations_c2w: np.ndarray,
        vio_camera_centers: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Returns (4x4 homogeneous Sim3 matrix, scalar_metric_scale).
        Applying this 4x4 matrix to mesh vertices, camera poses, and 3D Gaussians
        places the entire scene in Canonical RUB metric space with floor at Y=0.
        """
        # In OpenCV/COLMAP c2w, the camera's local -Y axis points UP in the world
        cam_up_vectors = -sfm_camera_rotations_c2w[:, :, 1]
        mean_up_hint = np.mean(cam_up_vectors, axis=0)

        floor_normal, floor_d = self.fit_ransac_floor_plane(
            np.asarray(mesh.vertices), sfm_camera_centers, mean_up_hint
        )

        # 1. Determine Metric Scale
        if vio_camera_centers is not None and len(vio_camera_centers) == len(sfm_camera_centers):
            scale, _, _ = self.umeyama_sim3(sfm_camera_centers, vio_camera_centers)
        else:
            cam_dists_to_floor = np.abs(sfm_camera_centers @ floor_normal + floor_d)
            median_unscaled_height = float(np.median(cam_dists_to_floor))
            scale = self.default_camera_height_m / max(median_unscaled_height, 1e-4)

        # 2. Rotate floor normal to canonical +Y [0, 1, 0]
        target_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        R_grav = self.rotation_matrix_from_vectors(floor_normal, target_up)

        # 3. Scale and rotate vertices to compute exact floor elevation and XZ center
        scaled_rotated_verts = scale * (np.asarray(mesh.vertices) @ R_grav.T)
        floor_y = float(np.percentile(scaled_rotated_verts[:, 1], 2.0))
        center_x = float(np.median(scaled_rotated_verts[:, 0]))
        center_z = float(np.median(scaled_rotated_verts[:, 2]))

        translation = np.array([-center_x, -floor_y, -center_z], dtype=np.float64)

        sim3_matrix = np.eye(4, dtype=np.float64)
        sim3_matrix[:3, :3] = scale * R_grav
        sim3_matrix[:3, 3] = translation

        return sim3_matrix, scale