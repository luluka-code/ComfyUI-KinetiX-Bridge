"""KinetiX 5120 维物理/语义残差桥核心模块。"""

from .bridge_model import BridgeConfig, WushuBridge, build_bridge, load_bridge, save_bridge
from .apply import apply_bridge_to_conditioning, magnitude_match, rms_normalize
from .dataset import PairDataset
from .pairs import build_pairs, dump_pairs
from .train import TrainConfig, train_bridge

__all__ = [
    "BridgeConfig",
    "WushuBridge",
    "build_bridge",
    "load_bridge",
    "save_bridge",
    "apply_bridge_to_conditioning",
    "magnitude_match",
    "rms_normalize",
    "PairDataset",
    "build_pairs",
    "dump_pairs",
    "TrainConfig",
    "train_bridge",
]
