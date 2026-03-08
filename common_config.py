from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent
COMMON_CONFIG_PATH = PROJECT_ROOT / "configs" / "common.yaml"


@lru_cache(maxsize=1)
def load_common_config() -> Dict[str, Any]:
    if not COMMON_CONFIG_PATH.exists():
        return {}
    cfg = OmegaConf.load(COMMON_CONFIG_PATH)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(resolved, dict):
        return resolved
    return {}


def get_common_path(key: str, default: Optional[str] = None) -> str:
    config = load_common_config()
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        return str(default) if default is not None else ""
    value = paths.get(key, default)
    return str(value) if value is not None else ""
