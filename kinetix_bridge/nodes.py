"""ComfyUI Pure Inference Node Implementation (Runtime Only).

Loads pre-trained 5120-d combat semantic bridge weights (.safetensors),
injecting authentic impact recoil and physical causality into action video generation
without fine-tuning the base diffusion model.
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
KINETIX_BRIDGE_DIR = os.path.join(MODELS_ROOT, "kinetix_bridge")
WUSHU_BRIDGE_DIR = os.path.join(MODELS_ROOT, "wushu_bridge")
LORAS_DIR = os.path.join(MODELS_ROOT, "loras")
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# 确保核心目录存在
os.makedirs(KINETIX_BRIDGE_DIR, exist_ok=True)
os.makedirs(WUSHU_BRIDGE_DIR, exist_ok=True)

if folder_paths is not None:
    for cat_name, p in [("kinetix_bridge", KINETIX_BRIDGE_DIR), ("wushu_bridge", WUSHU_BRIDGE_DIR)]:
        try:
            folder_paths.add_model_folder_path(cat_name, p)
        except Exception:
            pass

_BRIDGE_CACHE: Dict[Tuple[str, str], Any] = {}


def _list_weights() -> List[str]:
    """同时扫描 kinetix_bridge、wushu_bridge、loras 以及本地 models 目录中的 .safetensors 权重文件。"""
    files: List[str] = []
    
    # 1. 通过 ComfyUI 标准 folder_paths 接口扫描注册目录
    if folder_paths is not None:
        for cat in ("kinetix_bridge", "wushu_bridge", "loras"):
            try:
                flist = folder_paths.get_filename_list(cat)
                files.extend([f for f in flist if f.lower().endswith(".safetensors")])
            except Exception:
                pass

    # 2. 本地文件系统直接递归扫描（兼容未启动 ComfyUI 或相对路径/独立运行）
    search_dirs = [KINETIX_BRIDGE_DIR, WUSHU_BRIDGE_DIR, LORAS_DIR, LOCAL_MODELS_DIR]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for root, _, fnames in os.walk(sdir):
                for fname in fnames:
                    if fname.lower().endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, fname), sdir)
                        files.append(rel if rel != "." else fname)

    unique_files = sorted(list(set(files)))
    return unique_files if unique_files else ["none"]


def _resolve_weight(name: str) -> str:
    """按优先级顺序在多个模型目录中定位权重绝对路径。"""
    # 1. 优先通过 ComfyUI 标准系统路径解析
    if folder_paths is not None:
        for cat in ("kinetix_bridge", "wushu_bridge", "loras"):
            try:
                path = folder_paths.get_full_path(cat, name)
                if path and os.path.isfile(path):
                    return path
            except Exception:
                pass

    # 2. 在所有支持的目录中逐级搜索（含子目录）
    search_dirs = [KINETIX_BRIDGE_DIR, WUSHU_BRIDGE_DIR, LORAS_DIR, LOCAL_MODELS_DIR]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            # 直接匹配
            candidate = os.path.join(sdir, name)
            if os.path.isfile(candidate):
                return candidate
            # 递归匹配文件名
            for root, _, fnames in os.walk(sdir):
                if name in fnames:
                    return os.path.join(root, name)

    raise FileNotFoundError(
        f"Cannot find semantic bridge weight file: '{name}'.\n"
        f"Searched across multiple directories:\n"
        f"  1. {KINETIX_BRIDGE_DIR}\n"
        f"  2. {WUSHU_BRIDGE_DIR}\n"
        f"  3. {LORAS_DIR}\n"
        f"  4. {LOCAL_MODELS_DIR}\n"
        f"Please place your .safetensors bridge model in any of the above locations."
    )


def _pick_torch_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_bridge(name: str, device: str):
    if name == "none":
        raise ValueError(
            "Please select a valid semantic bridge weight file (.safetensors). "
            "If dropdown is empty, place your weights in ComfyUI/models/kinetix_bridge/ "
            "(or models/loras/, models/wushu_bridge/)."
        )
    key = (name, device)
    if key not in _BRIDGE_CACHE:
        path = _resolve_weight(name)
        _BRIDGE_CACHE[key] = load_bridge(path, device=device)
    return _BRIDGE_CACHE[key]


def _ui(result: tuple, *texts: str) -> dict:
    lines = [str(t) for t in texts if t is not None and str(t).strip()]
    return {"ui": {"text": lines}, "result": result}


# ══════════════════════════════════════════════════════════════════════
# 1. Core Inference Node: KinetiX Combat Semantic Bridge
# ══════════════════════════════════════════════════════════════════════

class KinetiXSemanticBridge:
    """Connects 5120-d combat residual bridge in between conditionings,

    outputting physically enhanced CONDITIONING with contact points,
    staggered recoil, and clashing sparks.
    Preserves all native metadata including minimax_keyframes and minimax_refs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bridge": (_list_weights(),),
                "alpha": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Injection strength: 0.18-0.25 for T2V; 0.10-0.12 for Ref2VA."}),
                "magnitude_match": (list(MAGNITUDE_MODES), {"default": "per_token",
                                                            "tooltip": "Magnitude matching mode: per_token preserves token energy and prevents color shifts."}),
                "token_span": (list(TEXT_SPAN_MODES), {"default": "all",
                                                       "tooltip": "all = full sequence; tail = modify only tail action tokens (crucial for Ref2VA to preserve face identity)."}),
                "tail_ratio": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                        "tooltip": "Tail sequence ratio when token_span is set to tail (default 0.5)."}),
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
    DESCRIPTION = "KinetiX 5120-d Semantic Bridge: Injects physical combat impact, weapon clashes, and staggered recoil into CONDITIONING space."

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
# 2. Auxiliary Tool Node: Clear Cache
# ══════════════════════════════════════════════════════════════════════

class KinetixClearCache:
    """Releases cached bridge models from RAM and VRAM."""

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
        msg = f"Released {n_b} cached semantic bridge model(s). VRAM cache cleared."
        return _ui((msg,), msg)


# ══════════════════════════════════════════════════════════════════════
# Node Mappings (Backwards compatible aliases preserved)
# ══════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    # Official Standard Names
    "KinetiXSemanticBridge": KinetiXSemanticBridge,
    "KinetixClearCache": KinetixClearCache,

    # Backwards Compatibility Aliases
    "H3WushuSemanticBridge": KinetiXSemanticBridge,
    "H3WushuClearCache": KinetixClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinetiXSemanticBridge": "KinetiX Combat Semantic Bridge",
    "KinetixClearCache": "KinetiX Clear Cache",

    "H3WushuSemanticBridge": "KinetiX Combat Semantic Bridge [Legacy]",
    "H3WushuClearCache": "KinetiX Clear Cache [Legacy]",
}
