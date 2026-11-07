"""ComfyUI 纯净核心节点映射。"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None

from .apply import MAGNITUDE_MODES, TEXT_SPAN_MODES, apply_bridge_to_conditioning
from .bridge_model import ARCH_MLP, ARCH_TRANS, load_bridge
from .dataset import PairDataset
from .judge import load_judge, score_conditioning
from .pairs import build_pairs, dump_pairs, pairs_stats

MODELS_ROOT = folder_paths.models_dir if folder_paths is not None else "models"
BRIDGE_DIR = os.path.join(MODELS_ROOT, "wushu_bridge")
DATASET_DIR = os.path.join(BRIDGE_DIR, "datasets")
os.makedirs(BRIDGE_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path("wushu_bridge", BRIDGE_DIR)
    except Exception:
        pass

_BRIDGE_CACHE: Dict[Tuple[str, str], Any] = {}
_JUDGE_CACHE: Dict[Tuple[str, str], Any] = {}


def _list_weights() -> List[str]:
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list("wushu_bridge")
            files = [f for f in files if f.lower().endswith(".safetensors")]
            if files:
                return sorted(files)
        except Exception:
            pass
    if os.path.isdir(BRIDGE_DIR):
        files = [f for f in os.listdir(BRIDGE_DIR) if f.lower().endswith(".safetensors")]
        if files:
            return sorted(files)
    return []


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
    raise FileNotFoundError(f"找不到权重文件：{name}（已在 {BRIDGE_DIR} 及 folder_paths 下搜索）")


def _detect_kind(name: str) -> str:
    low = name.lower()
    if "judge" in low or "jev" in low:
        return "judge"
    return "bridge"


def _bridge_choices() -> List[str]:
    files = [f for f in _list_weights() if _detect_kind(f) == "bridge"]
    return files if files else ["none"]


def _judge_choices() -> List[str]:
    files = [f for f in _list_weights() if _detect_kind(f) == "judge"]
    return ["none"] + files


def _pick_torch_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_bridge(name: str, device: str):
    if name == "none":
        raise ValueError("请先选择一个有效的桥权重文件（.safetensors）。")
    key = (name, device)
    if key not in _BRIDGE_CACHE:
        path = _resolve_weight(name)
        _BRIDGE_CACHE[key] = load_bridge(path, device=device)
    return _BRIDGE_CACHE[key]


def _get_judge(name: str, device: str):
    key = (name, device)
    if key not in _JUDGE_CACHE:
        path = _resolve_weight(name)
        _JUDGE_CACHE[key] = load_judge(path, device=device)
    return _JUDGE_CACHE[key]


def _ui(result: tuple, *texts: str) -> dict:
    lines = [str(t) for t in texts if t is not None and str(t).strip()]
    return {"ui": {"text": lines}, "result": result}


# ══════════════════════════════════════════════════════════════════════
# 1. 核心推理节点：KinetiX 武打语义逻辑桥
# ══════════════════════════════════════════════════════════════════════

class KinetiXSemanticBridge:
    """把 5120 维残差桥接在 conditioning 之间，输出物理修正后的 CONDITIONING。

    只替换 embedding 张量，minimax_keyframes / minimax_refs / minimax_frame_count
    等 metadata 原样保留，支持纯文本 T2V 与多图/首尾帧 Ref2VA。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "bridge": (_bridge_choices(),),
                "alpha": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "magnitude_match": (list(MAGNITUDE_MODES), {"default": "per_token"}),
                "token_span": (list(TEXT_SPAN_MODES), {"default": "all",
                                                       "tooltip": "all=全序列；tail=仅修改尾部动作描述（Ref2VA 强烈推荐 tail，保护参考图不毁脸）"}),
                "tail_ratio": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "chunk_tokens": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 256}),
                "allow_dim_mismatch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "judge": (_judge_choices(),),
                "auto_alpha": ("BOOLEAN", {"default": False,
                                           "tooltip": "用评分头自动调 alpha（需有效 judge 权重）"}),
                "auto_alpha_max": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guard": ("BOOLEAN", {"default": False,
                                      "tooltip": "改动后评分没变好就回退（慎用，可能因轻量模型敏感度误杀）"}),
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
        judge="none",
        auto_alpha=False,
        auto_alpha_max=0.25,
        guard=False,
    ):
        dev = _pick_torch_device(device)
        model = _get_bridge(bridge, dev)
        notes: List[str] = []

        judge_model = None
        score_before = None
        if judge and judge != "none":
            try:
                judge_model = _get_judge(judge, dev)
            except Exception as e:
                notes.append(f"[警告] 加载 judge 失败：{e}，已自动忽略打分。")

        if judge_model is not None:
            score_before, per = score_conditioning(judge_model, conditioning)
            notes.append(f"改动前武打逻辑分：{score_before:.3f}（每段 {[round(x,3) for x in per]}）")
            if auto_alpha:
                deficit = max(0.0, 0.75 - score_before) / 0.75
                alpha = float(min(auto_alpha_max, max(0.0, alpha + deficit * auto_alpha_max)))
                notes.append(f"自动 alpha -> {alpha:.3f}")

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

        final_score = 0.0
        if judge_model is not None:
            score_after, per_after = score_conditioning(judge_model, result)
            notes.append(f"改动后武打逻辑分：{score_after:.3f}（每段 {[round(x,3) for x in per_after]}）")
            if guard and score_before is not None and score_after < score_before - 1e-3:
                notes.append("守护开启：改动后分数下降，已回退为原 conditioning。")
                result = [[item[0], dict(item[1])] for item in conditioning]
                final_score = score_before
            else:
                final_score = score_after

        rep.messages = notes + rep.messages
        _txt = rep.to_text()
        return _ui((result, _txt, float(final_score)), _txt)


# ══════════════════════════════════════════════════════════════════════
# 2. 数据制作节点：KinetiX 构建训练对
# ══════════════════════════════════════════════════════════════════════

def _encode_texts(clip, texts: Sequence[str], max_tokens: int = 4096, progress=None) -> List[torch.Tensor]:
    embs: List[torch.Tensor] = []
    for i, t in enumerate(texts):
        tokens = clip.tokenize(t)
        cond = clip.encode_from_tokens(tokens, return_pooled=False)
        if isinstance(cond, (list, tuple)):
            tensor = cond[0][0] if isinstance(cond[0], (list, tuple)) else cond[0]
        else:
            tensor = cond
        if not torch.is_tensor(tensor):
            raise TypeError(f"期望张量，得到 {type(tensor)}")
        tensor = tensor.detach().cpu().to(torch.float16)
        if tensor.shape[1] > max_tokens:
            tensor = tensor[:, :max_tokens]
        embs.append(tensor)
        if progress is not None:
            progress(i + 1, len(texts))
    return embs


class KinetixBuildPairs:
    """从高质量动作正例文本出发，用 27 种物理算子自动合成负例并编码成 5120 维数据集。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "corpus_path": ("STRING", {"default": "datasets/kinetix_combat_pairs_all.jsonl"}),
                "out_path": ("STRING", {"default": "kinetix_combat_pairs_v1.npz"}),
                "max_pairs": ("INT", {"default": 800, "min": 1, "max": 100000}),
                "mode": (["t2v", "ref2v"], {"default": "t2v"}),
            },
            "optional": {
                "include_seeds": ("BOOLEAN", {"default": True}),
                "intensity_min": ("INT", {"default": 1, "min": 1, "max": 5}),
                "intensity_max": ("INT", {"default": 4, "min": 1, "max": 5}),
                "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "report")
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/KinetiX Bridge"
    OUTPUT_NODE = True

    def build(self, clip, corpus_path, out_path, max_pairs, mode,
              include_seeds=True, intensity_min=1, intensity_max=4, max_tokens=4096):
        out_path = out_path if os.path.isabs(out_path) else os.path.join(DATASET_DIR, out_path)
        if not out_path.endswith(".npz"):
            out_path += ".npz"

        # 智能解析语料路径
        cpath = corpus_path.strip()
        if not os.path.isabs(cpath) and not os.path.exists(cpath):
            candidates = [
                os.path.join(DATASET_DIR, cpath),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), cpath),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", os.path.basename(cpath)),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    cpath = cand
                    break

        pair_list, sources = build_pairs(
            corpus_path=cpath,
            include_seeds=bool(include_seeds),
            intensity_range=(int(intensity_min), int(intensity_max)),
            max_pairs=int(max_pairs),
        )
        if not pair_list:
            raise RuntimeError(f"未读取到任何训练对，请检查语料路径：{cpath}")

        texts: List[str] = []
        for p in pair_list:
            texts.append(p.bad)
            texts.append(p.good)

        print(f"[KinetiX-Bridge] 开始编码 {len(texts)} 条文本（mode={mode}）…")
        t0 = time.time()
        embs = _encode_texts(clip, texts, max_tokens=int(max_tokens))
        elapsed = time.time() - t0

        ds = PairDataset()
        ds.meta.mode = mode
        ds.meta.source = cpath
        ds.meta.encoder = type(clip).__name__

        import datetime
        ds.meta.created_at = datetime.datetime.now().isoformat(timespec="seconds")
        for i, p in enumerate(pair_list):
            ds.add(embs[2 * i], embs[2 * i + 1], p.bad_score, p.good_score, p.to_record())

        ds.save(out_path)
        dump_pairs(pair_list, os.path.splitext(out_path)[0] + "_pairs.jsonl")
        st = ds.stats()
        size_mb = os.path.getsize(out_path) / 1024 / 1024

        report = (
            f"数据集已成功保存：{out_path}\n"
            f"训练对：{st['count']} 对 ｜ 特征维度：{st['dim']} ｜ 模式：{mode}\n"
            f"文件大小：{size_mb:.1f} MB ｜ 编码耗时：{elapsed:.0f}s\n"
            f"负例均分：{st['x_score_mean']} ｜ 正例均分：{st['y_score_mean']}"
        )
        print("[KinetiX-Bridge] " + report.replace("\n", " | "))
        return _ui((out_path, report), report)


# ══════════════════════════════════════════════════════════════════════
# 3. 训练节点：KinetiX 训练残差桥
# ══════════════════════════════════════════════════════════════════════

class KinetixTrainBridge:
    """1 键训练 5120 维物理残差桥，带早停保护与序列截断防爆显存。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "kinetix_combat_pairs_v1.npz"}),
                "arch": ([ARCH_TRANS, ARCH_MLP], {"default": ARCH_TRANS}),
                "out_name": ("STRING", {"default": "kinetix_combat_bridge_v1.safetensors"}),
                "epochs": ("INT", {"default": 60, "min": 1, "max": 2000}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 512}),
                "lr": ("FLOAT", {"default": 0.0003, "min": 1e-6, "max": 0.1, "step": 1e-5}),
                "hidden": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "layers": ("INT", {"default": 2, "min": 1, "max": 12}),
                "heads": ("INT", {"default": 4, "min": 1, "max": 32}),
                "anchor_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01}),
                "alpha_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "val_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "mode": (["t2v", "ref2v", "any"], {"default": "t2v"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
                "max_seq_tokens": ("INT", {"default": 2048, "min": 32, "max": 65536, "step": 32}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("bridge_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/KinetiX Bridge"
    OUTPUT_NODE = True

    def run(self, dataset_path, arch, out_name, epochs, batch_size, lr, hidden, layers, heads,
            anchor_weight, alpha_min, alpha_max, val_ratio, device,
            mode="t2v", seed=1234, max_seq_tokens=2048):
        from .train import TrainConfig, train_bridge

        path = dataset_path if os.path.isabs(dataset_path) else os.path.join(DATASET_DIR, dataset_path)
        ds = PairDataset.load(path)
        out = out_name if os.path.isabs(out_name) else os.path.join(BRIDGE_DIR, out_name)
        if not out.endswith(".safetensors"):
            out += ".safetensors"

        cfg = TrainConfig(
            arch=arch, hidden=int(hidden), layers=int(layers), heads=int(heads),
            mode=mode, name=os.path.splitext(os.path.basename(out))[0],
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            anchor_weight=float(anchor_weight), alpha_min=float(alpha_min), alpha_max=float(alpha_max),
            val_ratio=float(val_ratio), seed=int(seed), device=device,
            max_seq_tokens=int(max_seq_tokens),
        )
        report = train_bridge(ds, cfg, out, log=lambda s: print("[KinetiX-Bridge] " + str(s)))
        text = (
            f"模型已保存：{out}\n"
            f"架构：{report['arch']} ｜ 参数量：{report['params']:,}（{report['size_mb_fp32']} MB）\n"
            f"样本量：{report['samples']}（训练 {report['train_samples']} / 验证 {report['val_samples']}）\n"
            f"训练耗时：{report['seconds']}s ｜ 收敛状态：已成功收敛"
        )
        return _ui((out, text), text)


# ══════════════════════════════════════════════════════════════════════
# 4. 辅助工具节点：清理显存与缓存
# ══════════════════════════════════════════════════════════════════════

class KinetixClearCache:
    """清理内存/显存中缓存的语义桥模型与中间张量。"""

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
        n_j = len(_JUDGE_CACHE)
        _BRIDGE_CACHE.clear()
        _JUDGE_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        msg = f"已释放缓存：桥模型 {n_b} 个，裁判模型 {n_j} 个，已清理显存。"
        return _ui((msg,), msg)


class KinetixTrainJevHead:
    """训练 JEV 式评分头（可选进阶工具）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "kinetix_combat_pairs_v1.npz"}),
                "out_name": ("STRING", {"default": "kinetix_combat_jev_v1.safetensors"}),
                "hidden": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                "epochs": ("INT", {"default": 80, "min": 1, "max": 2000}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 4096}),
                "lr": ("FLOAT", {"default": 0.001, "min": 1e-6, "max": 0.1, "step": 1e-5}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
            "optional": {
                "heads": ("INT", {"default": 4, "min": 1, "max": 32}),
                "dropout": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.9, "step": 0.01}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("judge_path", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/KinetiX Bridge"
    OUTPUT_NODE = True

    def run(self, dataset_path, out_name, hidden, epochs, batch_size, lr, device,
            heads=4, dropout=0.05, seed=1234):
        from .train import JudgeTrainConfig, train_judge

        path = dataset_path if os.path.isabs(dataset_path) else os.path.join(DATASET_DIR, dataset_path)
        ds = PairDataset.load(path)
        out = out_name if os.path.isabs(out_name) else os.path.join(BRIDGE_DIR, out_name)
        if not out.endswith(".safetensors"):
            out += ".safetensors"

        cfg = JudgeTrainConfig(
            hidden=int(hidden), heads=int(heads), dropout=float(dropout),
            name=os.path.splitext(os.path.basename(out))[0],
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            seed=int(seed), device=device,
        )
        report = train_judge(ds, cfg, out, log=lambda s: print("[KinetiX-Bridge] " + str(s)))
        text = f"裁判头已导出：{out}\n参数量：{report['params']:,} ｜ 验证准确率：{report['val_accuracy']}"
        return _ui((out, text), text)


# ══════════════════════════════════════════════════════════════════════
# 节点映射表（同时注册新名称与旧名称，保障 100% 向后兼容不红节点）
# ══════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    # 官方标准新命名
    "KinetiXSemanticBridge": KinetiXSemanticBridge,
    "KinetixBuildPairs": KinetixBuildPairs,
    "KinetixTrainBridge": KinetixTrainBridge,
    "KinetixTrainJevHead": KinetixTrainJevHead,
    "KinetixClearCache": KinetixClearCache,

    # 向后兼容别名（打开老工作流时不红节点）
    "H3WushuSemanticBridge": KinetiXSemanticBridge,
    "H3WushuBuildDataset": KinetixBuildPairs,
    "H3WushuBuildPairs": KinetixBuildPairs,
    "H3WushuTrainBridge": KinetixTrainBridge,
    "H3WushuTrainJevHead": KinetixTrainJevHead,
    "H3WushuClearCache": KinetixClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinetiXSemanticBridge": "KinetiX 武打语义逻辑桥 (Combat Semantic Bridge)",
    "KinetixBuildPairs": "KinetiX 构建训练对 (Build Pairs)",
    "KinetixTrainBridge": "KinetiX 训练残差桥 (Train Bridge)",
    "KinetixTrainJevHead": "KinetiX 训练评分头 (Train Judge)",
    "KinetixClearCache": "KinetiX 清理缓存 (Clear Cache)",

    "H3WushuSemanticBridge": "KinetiX 武打语义逻辑桥 [Legacy]",
    "H3WushuBuildDataset": "KinetiX 构建训练对 [Legacy]",
    "H3WushuBuildPairs": "KinetiX 构建训练对 [Legacy]",
    "H3WushuTrainBridge": "KinetiX 训练残差桥 [Legacy]",
    "H3WushuTrainJevHead": "KinetiX 训练评分头 [Legacy]",
    "H3WushuClearCache": "KinetiX 清理缓存 [Legacy]",
}
