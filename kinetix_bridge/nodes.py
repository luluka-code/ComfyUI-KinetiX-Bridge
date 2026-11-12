"""ComfyUI Pure Inference Node Implementation (Runtime Only).

Loads pre-trained 5120-d combat semantic bridge weights (.safetensors),
injecting authentic impact recoil and physical causality into action video generation
without fine-tuning the base diffusion model.

Supports:
1. Automatic downloading from Hugging Face (kinetixfx/Kinetix_Combat_Bridge) & domestic mirror (hf-mirror.com).
2. Manual on-disk weight scanning across multiple directories (models/kinetix_bridge, models/loras, etc.).
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

# 官方 Hugging Face 权重元数据
HF_REPO_ID = "kinetixfx/Kinetix_Combat_Bridge"
HF_FILENAME_V1_1 = "kinetix_combat_bridge_v1.1.safetensors"
HF_FILENAME_V1 = "kinetix_combat_bridge_v1.safetensors"
AUTO_DOWNLOAD_LABEL = f"{HF_FILENAME_V1_1} (Auto Download)"
AUTO_DOWNLOAD_LABEL_V1 = f"{HF_FILENAME_V1} (Auto Download)"


def _download_official_model(repo_id: str = HF_REPO_ID, filename: str = HF_FILENAME_V1_1) -> str:
    """自动下载官方权重至 models/kinetix_bridge/ 目录。支持 Hugging Face 官方源与国内高速镜像双通道容错。"""
    os.makedirs(KINETIX_BRIDGE_DIR, exist_ok=True)
    target_path = os.path.join(KINETIX_BRIDGE_DIR, filename)

    if os.path.isfile(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        return target_path

    print(f"\n[KinetiX Bridge] Official model '{filename}' not found locally.")
    print(f"[KinetiX Bridge] Starting automatic download from Hugging Face: {repo_id}...")

    # 通道 1: 尝试 huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=KINETIX_BRIDGE_DIR,
            local_dir_use_symlinks=False,
        )
        if os.path.isfile(downloaded) and os.path.getsize(downloaded) > 1024 * 1024:
            print(f"[KinetiX Bridge] Successfully downloaded via huggingface_hub to: {downloaded}")
            return downloaded
    except Exception as e:
        print(f"[KinetiX Bridge] huggingface_hub download attempt failed ({e}), switching to direct stream...")

    # 通道 2 & 3: torch.hub 流式下载（依次尝试自定义 HF_ENDPOINT、官方源与国内高速镜像 hf-mirror.com）
    endpoints = []
    custom_ep = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if custom_ep:
        endpoints.append(custom_ep)
    endpoints.extend(["https://huggingface.co", "https://hf-mirror.com"])

    last_err = None
    for ep in endpoints:
        url = f"{ep}/{repo_id}/resolve/main/{filename}"
        try:
            print(f"[KinetiX Bridge] Connecting to: {url} ...")
            torch.hub.download_url_to_file(url, target_path, progress=True)
            if os.path.isfile(target_path) and os.path.getsize(target_path) > 1024 * 1024:
                print(f"[KinetiX Bridge] Successfully downloaded model to {target_path}")
                return target_path
        except Exception as err:
            last_err = err
            print(f"[KinetiX Bridge] Download attempt from {ep} failed: {err}. Trying next mirror...")

    # 若全部通道失败，提供清晰的手动下载指南
    raise RuntimeError(
        f"[KinetiX Bridge] Failed to automatically download '{filename}' from Hugging Face.\n"
        f"Error details: {last_err}\n\n"
        f"Manual Download Instructions:\n"
        f"  1. Download '{filename}' from:\n"
        f"     https://huggingface.co/{repo_id}/blob/main/{filename}\n"
        f"     (or Mirror: https://hf-mirror.com/{repo_id}/blob/main/{filename})\n"
        f"  2. Place the file into:\n"
        f"     {KINETIX_BRIDGE_DIR}/\n"
    )


def _list_weights() -> List[str]:
    """同时提供官方云端自动下载项与本地已下载/手动放置的权重扫描。"""
    models: List[str] = [AUTO_DOWNLOAD_LABEL, AUTO_DOWNLOAD_LABEL_V1]

    # 1. 扫描本地各目录
    on_disk: List[str] = []
    if folder_paths is not None:
        for cat in ("kinetix_bridge", "wushu_bridge", "loras"):
            try:
                flist = folder_paths.get_filename_list(cat)
                on_disk.extend([f for f in flist if f.lower().endswith(".safetensors")])
            except Exception:
                pass

    search_dirs = [KINETIX_BRIDGE_DIR, WUSHU_BRIDGE_DIR, LORAS_DIR, LOCAL_MODELS_DIR]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for root, _, fnames in os.walk(sdir):
                for fname in fnames:
                    if fname.lower().endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, fname), sdir)
                        on_disk.append(rel if rel != "." else fname)

    unique_on_disk = sorted(list(set(on_disk)))
    for item in unique_on_disk:
        display_name = f"{item} (on disk)"
        if display_name not in models:
            models.append(display_name)

    return models


def _resolve_weight(name: str) -> str:
    """按优先级顺序在多个模型目录中定位权重绝对路径，支持自动下载。"""
    clean_name = name

    # 1. 如果选择了 "(Auto Download)"，直接触发自动下载并返回下载路径
    if clean_name.endswith(" (Auto Download)"):
        clean_name = clean_name.replace(" (Auto Download)", "").strip()
        return _download_official_model(HF_REPO_ID, clean_name or HF_FILENAME_V1_1)

    # 2. 如果是 "(on disk)" 本地选项，剥离后缀后查找
    if clean_name.endswith(" (on disk)"):
        clean_name = clean_name.replace(" (on disk)", "").strip()

    # 3. 优先通过 ComfyUI 标准系统路径解析
    if folder_paths is not None:
        for cat in ("kinetix_bridge", "wushu_bridge", "loras"):
            try:
                path = folder_paths.get_full_path(cat, clean_name)
                if path and os.path.isfile(path):
                    return path
            except Exception:
                pass

    # 4. 在所有支持的目录中逐级搜索（含子目录）
    search_dirs = [KINETIX_BRIDGE_DIR, WUSHU_BRIDGE_DIR, LORAS_DIR, LOCAL_MODELS_DIR]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            candidate = os.path.join(sdir, clean_name)
            if os.path.isfile(candidate):
                return candidate
            for root, _, fnames in os.walk(sdir):
                if clean_name in fnames:
                    return os.path.join(root, clean_name)

    # 5. 容错：如果用户选了官方模型名字但本地不存在，自动下载拯救
    if "kinetix_combat_bridge" in clean_name.lower():
        return _download_official_model(HF_REPO_ID, HF_FILENAME)

    raise FileNotFoundError(
        f"Cannot find semantic bridge weight file: '{clean_name}'.\n"
        f"Searched across multiple directories:\n"
        f"  1. {KINETIX_BRIDGE_DIR}\n"
        f"  2. {WUSHU_BRIDGE_DIR}\n"
        f"  3. {LORAS_DIR}\n"
        f"  4. {LOCAL_MODELS_DIR}\n"
        f"Please select '{AUTO_DOWNLOAD_LABEL}' to auto-download from Hugging Face,\n"
        f"or manually place '{clean_name}' into any of the above locations."
    )


def _pick_torch_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_bridge(name: str, device: str):
    if name == "none":
        raise ValueError(
            "Please select a valid semantic bridge weight file (.safetensors). "
            f"Select '{AUTO_DOWNLOAD_LABEL}' to automatically download from Hugging Face."
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
    Supports auto-downloading from Hugging Face and local on-disk scanning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bridge": (_list_weights(), {
                    "default": AUTO_DOWNLOAD_LABEL,
                    "tooltip": "Select semantic bridge model. '(Auto Download)' automatically downloads from HuggingFace (kinetixfx/Kinetix_Combat_Bridge) to models/kinetix_bridge/. '(on disk)' uses local weights."
                }),
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
# Node Mappings
# ══════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "KinetiXSemanticBridge": KinetiXSemanticBridge,
    "KinetixClearCache": KinetixClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinetiXSemanticBridge": "KinetiX Combat Semantic Bridge",
    "KinetixClearCache": "KinetiX Clear Cache",
}
