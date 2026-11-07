"""ComfyUI 纯推理插件核心节点（Runtime Only）。

专职负责加载预训练的 5120 维物理/动作语义桥（.safetensors），
在无需重训大模型的前提下，为打斗视频注入真实打击感与物理因果。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None

from .apply import MAGNITUDE_MODES, TEXT_SPAN_MODES, apply_bridge_to_conditioning
from .bridge_model import load_bridge

MODELS_ROOT = folder_paths.models_dir if folder_paths is not None else "models"
BRIDGE_DIR = os.path.join(MODELS_ROOT, "wushu_bridge")
os.makedirs(BRIDGE_DIR, exist_ok=True)

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path("wushu_bridge", BRIDGE_DIR)
    except Exception:
        pass

_BRIDGE_CACHE: Dict[Tuple[str, str], Any] = {}


def _list_weights() -> List[str]:
    files = []
    if folder_paths is not None:
        try:
            flist = folder_paths.get_filename_list("wushu_bridge")
            files.extend([f for f in flist if f.lower().endswith(".safetensors")])
        except Exception:
            pass
    if os.path.isdir(BRIDGE_DIR):
        files.extend([f for f in os.listdir(BRIDGE_DIR) if f.lower().endswith(".safetensors")])

    # 当前插件内预置的 models/ 目录作为备用扫描路径
    local_models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if os.path.isdir(local_models):
        files.extend([f for f in os.listdir(local_models) if f.lower().endswith(".safetensors")])

    unique_files = sorted(list(set(files)))
    return unique_files if unique_files else ["none"]


def _resolve_weight(name: str) -> str:
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("wushu_bridge", name)
            if path and os.path.isfile(path):
                return path
        except Exception:
            pass

    direct = os.path.join(BRIDGE_DIR, name)
    if os.path.isfile(direct):
        return direct

    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", name)
    if os.path.isfile(local):
        return local

    raise FileNotFoundError(f"找不到语义桥权重文件：{name}（请将 .safetensors 放入 {BRIDGE_DIR}）")


def _pick_torch_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_bridge(name: str, device: str):
    if name == "none":
        raise ValueError("请先选择一个有效的语义桥权重文件（.safetensors）。若下拉为空，请将权重放入 ComfyUI/models/wushu_bridge/ 目录。")
    key = (name, device)
    if key not in _BRIDGE_CACHE:
        path = _resolve_weight(name)
        _BRIDGE_CACHE[key] = load_bridge(path, device=device)
    return _BRIDGE_CACHE[key]


def _ui(result: tuple, *texts: str) -> dict:
    lines = [str(t) for t in texts if t is not None and str(t).strip()]
    return {"ui": {"text": lines}, "result": result}


# ══════════════════════════════════════════════════════════════════════
# 1. 核心推理节点：KinetiX 武打语义逻辑桥 (Combat Semantic Bridge)
# ══════════════════════════════════════════════════════════════════════

class KinetiXSemanticBridge:
    """把 5120 维打斗残差桥接在 conditioning 之间，输出物理修正后的 CONDITIONING。

    只替换 embedding 张量，minimax_keyframes / minimax_refs / minimax_frame_count
    等 metadata 原样保留，支持纯文本 T2V 与多图/首尾帧 Ref2VA。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bridge": (_list_weights(),),
                "alpha": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "注入强度：T2V 推荐 0.18~0.25；Ref2VA 推荐 0.10~0.12"}),
                "magnitude_match": (list(MAGNITUDE_MODES), {"default": "per_token",
                                                            "tooltip": "模长对齐模式：推荐 per_token，保护特征能量恒定防偏色"}),
                "token_span": (list(TEXT_SPAN_MODES), {"default": "all",
                                                       "tooltip": "all=全序列；tail=仅微调尾部动作词（Ref2VA 图生视频必须选 tail，保护人脸不走样）"}),
                "tail_ratio": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                        "tooltip": "tail 模式下生效的尾部比例，默认 0.5"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "chunk_tokens": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 256}),
                "allow_dim_mismatch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "FLOAT")
    RETURN_NAMES = ("conditioning", "report", "logic_score")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/KinetiX Bridge"
    OUTPUT_NODE = True
    DESCRIPTION = "KinetiX 5120 维语义桥：在条件空间对 CONDITIONING 注入真实打击感与物理因果残差"

    def apply(
        self,
        conditioning,
        bridge,
        alpha,
        magnitude_match,
        token_span,
        tail_ratio,
        device,
        chunk_tokens,
        allow_dim_mismatch,
    ):
        dev = _pick_torch_device(device)
        model = _get_bridge(bridge, dev)

        result, rep = apply_bridge_to_conditioning(
            conditioning,
            model,
            alpha=float(alpha),
            magnitude_match_mode=magnitude_match,
            token_span=token_span,
            tail_ratio=float(tail_ratio),
            max_tokens_per_chunk=int(chunk_tokens),
            allow_dim_mismatch=bool(allow_dim_mismatch),
        )

        _txt = rep.to_text()
        return _ui((result, _txt, 1.0), _txt)


# ══════════════════════════════════════════════════════════════════════
# 2. 辅助工具节点：清理显存与缓存
# ══════════════════════════════════════════════════════════════════════

class KinetixClearCache:
    """清理内存/显存中缓存的语义桥模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "MiniMax H3/KinetiX Bridge"
    OUTPUT_NODE = True

    def clear(self):
        n_b = len(_BRIDGE_CACHE)
        _BRIDGE_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        msg = f"已释放 {n_b} 个语义桥模型缓存，显存已清理。"
        return _ui((msg,), msg)


# ══════════════════════════════════════════════════════════════════════
# 节点映射表（保留向后兼容别名，打开老工作流时不红节点）
# ══════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    # 官方标准新命名
    "KinetiXSemanticBridge": KinetiXSemanticBridge,
    "KinetixClearCache": KinetixClearCache,

    # 向后兼容别名（打开旧工作流时不报错）
    "H3WushuSemanticBridge": KinetiXSemanticBridge,
    "H3WushuClearCache": KinetixClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinetiXSemanticBridge": "KinetiX 武打语义逻辑桥 (Combat Semantic Bridge)",
    "KinetixClearCache": "KinetiX 清理缓存 (Clear Cache)",

    "H3WushuSemanticBridge": "KinetiX 武打语义逻辑桥 [Legacy]",
    "H3WushuClearCache": "KinetiX 清理缓存 [Legacy]",
}
