from pathlib import Path

# Root data directory for all scenes
DATA_ROOT = Path("./data/scenes")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Default reconstruction parameters
DEFAULT_CAMERA_HEIGHT_M = 1.45
DEFAULT_AGENT_RADIUS_M = 0.28
DEFAULT_CELL_SIZE_M = 0.05
DEFAULT_VOXEL_SIZE_M = 0.06