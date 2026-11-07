"""KinetiX 5120 维物理打击感与动作语义桥（纯推理运行时）。"""

from .bridge_model import BridgeConfig, WushuBridge, build_bridge, load_bridge
from .apply import apply_bridge_to_conditioning, magnitude_match, rms_normalize

__all__ = [
    "BridgeConfig",
    "WushuBridge",
    "build_bridge",
    "load_bridge",
    "apply_bridge_to_conditioning",
    "magnitude_match",
    "rms_normalize",
]
