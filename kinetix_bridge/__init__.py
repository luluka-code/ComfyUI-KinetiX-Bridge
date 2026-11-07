"""KinetiX 5120-d Combat Semantic Bridge (Inference Runtime)."""

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
