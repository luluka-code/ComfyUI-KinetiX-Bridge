"""ComfyUI-KinetiX-Bridge: MiniMax-H3 5120-d Combat Semantic Bridge.

专为 MiniMax-H3 动作视频打造的 5120 维物理因果与打击感语义残差桥。
在文本/参考图条件空间注入真实物理受击、武器碰撞与动量传递因果。
"""

from .kinetix_bridge.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.0.0"
WEB_DIRECTORY = None

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
