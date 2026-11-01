import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree


@dataclass
class SpatialPOI:
    poi_id: str
    label: str
    centroid_xyz: List[float]
    bbox_min_xyz: List[float]
    bbox_max_xyz: List[float]
    confidence: float
    embedding: List[float]


class VoxelSemanticField:
    """
    Sparse 3D Voxel Grid that fuses multi-view 2D vision-language (CLIP) embeddings
    into persistent 3D geometric space using depth back-projection and DBSCAN clustering.
    """

    def __init__(self, voxel_size_m: float = 0.06, feature_dim: int = 512):
        self.voxel_size_m = voxel_size_m
        self.feature_dim = feature_dim
        self.voxel_features: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.voxel_weights: Dict[Tuple[int, int, int], float] = {}
        self.saved_pois: List[SpatialPOI] = []

    def _points_to_voxel_keys(self, points_xyz: np.ndarray) -> np.ndarray:
        return np.floor(points_xyz / self.voxel_size_m).astype(np.int32)

    def fuse_mask_observation(
        self,
        mask_2d: np.ndarray,
        depth_map_m: np.ndarray,
        intrinsics_3x3: np.ndarray,
        c2w_4x4: np.ndarray,
        clip_embedding: np.ndarray,
        min_depth_m: float = 0.25,
        max_depth_m: float = 8.0,
    ) -> None:
        """
        Back-projects all valid pixels inside a 2D segmentation mask into 3D canonical space
        and accumulates the normalized CLIP feature vector into intersected sparse voxels.
        """
        norm_emb = clip_embedding / (np.linalg.norm(clip_embedding) + 1e-8)
        vs, us = np.nonzero(mask_2d)
        if len(us) == 0:
            return

        # Subsample dense masks for fast processing
        if len(us) > 4000:
            idx = np.random.choice(len(us), size=4000, replace=False)
            us, vs = us[idx], vs[idx]

        depths = depth_map_m[vs, us]
        valid = (depths >= min_depth_m) & (depths <= max_depth_m) & np.isfinite(depths)
        if np.sum(valid) < 15:
            return

        us, vs, depths = us[valid], vs[valid], depths[valid]

        fx, fy = intrinsics_3x3[0, 0], intrinsics_3x3[1, 1]
        cx, cy = intrinsics_3x3[0, 2], intrinsics_3x3[1, 2]

        # Back-project into camera frame (OpenCV convention: +X Right, +Y Down, +Z Forward)
        x_cam = (us - cx) * depths / fx
        y_cam = (vs - cy) * depths / fy
        z_cam = depths
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)  # (N, 4)

        # Transform to Canonical RUB World Space
        pts_world = (c2w_4x4 @ pts_cam.T).T[:, :3]

        # Reject statistical depth outliers (flying pixels at occlusion boundaries)
        if len(pts_world) > 30:
            med = np.median(pts_world, axis=0)
            dists = np.linalg.norm(pts_world - med, axis=1)
            inliers = dists <= np.percentile(dists, 90.0)
            pts_world = pts_world[inliers]
            depths = depths[inliers]

        voxel_coords = self._points_to_voxel_keys(pts_world)
        unique_voxels, inverse_idx = np.unique(voxel_coords, axis=0, return_inverse=True)

        for idx_v, v_coord in enumerate(unique_voxels):
            key = (int(v_coord[0]), int(v_coord[1]), int(v_coord[2]))
            mean_d = float(np.mean(depths[inverse_idx == idx_v]))
            weight = 1.0 / (1.0 + 0.25 * (mean_d ** 2))

            if key not in self.voxel_features:
                self.voxel_features[key] = norm_emb * weight
                self.voxel_weights[key] = weight
            else:
                self.voxel_features[key] += norm_emb * weight
                self.voxel_weights[key] += weight

    def _cluster_active_voxels(
        self, coords_xyz: np.ndarray, scores: np.ndarray, eps_m: float = 0.22, min_samples: int = 5
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Pure SciPy cKDTree connected-components implementation of 3D DBSCAN
        so there are zero fragile C-extension version conflicts.
        """
        if len(coords_xyz) < min_samples:
            return []

        tree = cKDTree(coords_xyz)
        neighbors = tree.query_ball_tree(tree, r=eps_m)
        is_core = np.array([len(n) >= min_samples for n in neighbors])

        visited = np.zeros(len(coords_xyz), dtype=bool)
        clusters: List[Tuple[np.ndarray, float]] = []

        for i in range(len(coords_xyz)):
            if visited[i] or not is_core[i]:
                continue
            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                cluster_indices.append(curr)
                if is_core[curr]:
                    for nb in neighbors[curr]:
                        if not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)

            if len(cluster_indices) >= min_samples:
                c_pts = coords_xyz[cluster_indices]
                c_score = float(np.mean(scores[cluster_indices]))
                clusters.append((c_pts, c_score))

        clusters.sort(key=lambda x: x[1], reverse=True)
        return clusters

    def query_open_vocabulary(
        self,
        text_embedding: np.ndarray,
        query_label: str,
        similarity_threshold: float = 0.26,
        top_k: int = 5,
    ) -> List[SpatialPOI]:
        """
        Queries the 3D semantic voxel field with a CLIP text embedding and returns
        clustered 3D bounding boxes and centroids sorted by semantic relevance.
        """
        if not self.voxel_features:
            return []

        q_emb = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        keys = list(self.voxel_features.keys())
        raw_feats = np.array([self.voxel_features[k] for k in keys])
        norms = np.linalg.norm(raw_feats, axis=1, keepdims=True) + 1e-8
        unit_feats = raw_feats / norms

        similarities = unit_feats @ q_emb
        active_mask = similarities >= similarity_threshold
        if np.sum(active_mask) == 0:
            return []

        active_keys = np.array(keys)[active_mask]
        active_xyz = (active_keys.astype(np.float64) + 0.5) * self.voxel_size_m
        active_scores = similarities[active_mask]

        clusters = self._cluster_active_voxels(active_xyz, active_scores)
        results: List[SpatialPOI] = []

        for idx, (pts, score) in enumerate(clusters[:top_k]):
            bbox_min = np.min(pts, axis=0) - (0.5 * self.voxel_size_m)
            bbox_max = np.max(pts, axis=0) + (0.5 * self.voxel_size_m)
            centroid = np.mean(pts, axis=0)

            results.append(
                SpatialPOI(
                    poi_id=f"poi_{query_label.replace(' ', '_')}_{idx}",
                    label=query_label,
                    centroid_xyz=np.round(centroid, 3).tolist(),
                    bbox_min_xyz=np.round(bbox_min, 3).tolist(),
                    bbox_max_xyz=np.round(bbox_max, 3).tolist(),
                    confidence=round(score, 4),
                    embedding=np.round(q_emb, 4).tolist(),
                )
            )

        return results

    def save(self, filepath: Path) -> None:
        serializable = {
            "voxel_size_m": self.voxel_size_m,
            "voxels": [
                {
                    "key": list(k),
                    "weight": float(self.voxel_weights[k]),
                    "feat": np.round(
                        v / (np.linalg.norm(v) + 1e-8), 4
                    ).tolist(),
                }
                for k, v in self.voxel_features.items()
                if self.voxel_weights[k] > 0.35
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f)