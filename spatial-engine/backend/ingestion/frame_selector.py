import cv2
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionPipeline")


@dataclass
class KeyframeMetadata:
    frame_idx: int
    timestamp_sec: float
    filename: str
    sharpness_score: float
    mean_optical_flow: float
    arkit_transform: Optional[List[List[float]]] = None


class SmartFrameSelector:
    """
    Extracts sharp, geometrically well-distributed keyframes from raw smartphone video.
    Uses sliding-window Laplacian variance filtering coupled with Lucas-Kanade optical
    flow parallax gating to prevent degenerate SfM camera baselines.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_sharpness: float = 85.0,
        min_parallax_px: float = 28.0,
        max_parallax_px: float = 160.0,
        target_max_dim: int = 1600,
    ):
        self.window_size = window_size
        self.min_sharpness = min_sharpness
        self.min_parallax_px = min_parallax_px
        self.max_parallax_px = max_parallax_px
        self.target_max_dim = target_max_dim

    @staticmethod
    def compute_sharpness(gray_img: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def compute_parallax(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=12
        )
        if pts_prev is None or len(pts_prev) < 20:
            return 999.0  # Force keyframe if texture is extremely low

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_prev, None, winSize=(21, 21), maxLevel=3
        )
        valid = status.flatten() == 1
        if np.sum(valid) < 12:
            return 999.0

        displacements = np.linalg.norm(pts_curr[valid] - pts_prev[valid], axis=2)
        return float(np.median(displacements))

    def _resize_preserve_aspect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= self.target_max_dim:
            return img
        scale = self.target_max_dim / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        arkit_poses_path: Optional[Path] = None,
    ) -> List[KeyframeMetadata]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        arkit_data: Dict[int, List[List[float]]] = {}
        if arkit_poses_path and arkit_poses_path.exists():
            with open(arkit_poses_path, "r", encoding="utf-8") as f:
                raw_poses = json.load(f)
                for item in raw_poses.get("frames", []):
                    arkit_data[int(item["frame_idx"])] = item["transform_matrix"]

        window_buffer = []
        selected_keyframes: List[KeyframeMetadata] = []
        last_keyframe_gray: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = self._resize_preserve_aspect(frame)
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            sharpness = self.compute_sharpness(gray)

            window_buffer.append((frame_idx, resized_frame, gray, sharpness))

            if len(window_buffer) >= self.window_size:
                # Pick sharpest candidate within temporal window
                best_candidate = max(window_buffer, key=lambda x: x[3])
                c_idx, c_frame, c_gray, c_sharp = best_candidate

                if c_sharp >= self.min_sharpness or len(selected_keyframes) == 0:
                    parallax = (
                        0.0
                        if last_keyframe_gray is None
                        else self.compute_parallax(last_keyframe_gray, c_gray)
                    )

                    if last_keyframe_gray is None or parallax >= self.min_parallax_px:
                        filename = f"frame_{c_idx:06d}.jpg"
                        out_file = output_dir / filename
                        cv2.imwrite(
                            str(out_file), c_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        )

                        meta = KeyframeMetadata(
                            frame_idx=c_idx,
                            timestamp_sec=round(c_idx / fps, 4),
                            filename=filename,
                            sharpness_score=round(c_sharp, 2),
                            mean_optical_flow=round(parallax, 2),
                            arkit_transform=arkit_data.get(c_idx),
                        )
                        selected_keyframes.append(meta)
                        last_keyframe_gray = c_gray

                window_buffer.clear()

            frame_idx += 1

        cap.release()
        manifest_path = output_dir / "keyframes_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in selected_keyframes], f, indent=2)

        logger.info(
            f"Extracted {len(selected_keyframes)} sharp keyframes from {frame_idx} total frames."
        )
        return selected_keyframes