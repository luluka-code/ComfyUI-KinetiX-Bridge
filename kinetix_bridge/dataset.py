"""训练数据集的磁盘格式（.npz + 同名 .json 元信息）。

为什么要自己定格式
------------------
训练桥需要的是 **H3 文本编码器输出的 5120 维 token 序列**。这只能在装了
H3 的 ComfyUI 里跑（外部设备跑不动那个编码器）。所以流程是：

1. 在 ComfyUI 里用 ``H3 Wushu Build Dataset`` 节点把 (负例文本, 正例文本)
   成对编码，落盘成 ``*_pairs.npz``；
2. 再用 ``H3 Wushu Train Bridge`` 节点训练并导出 safetensors；
3. 之后推理时就不再需要编码器了（桥只吃 conditioning）。

为了让 924 条语料在 5120 维下不至于爆盘，序列用变长紧凑存储：
``x_flat / x_len`` 拼在一起，token 用 float16。

同时存一份**池化向量**给评分头用（``x_pool / y_pool``，float32），
以及每条样本的规则分（``x_score / y_score``，0~1 软标签）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PairMeta:
    kind: str = "wushu_bridge_pairs"
    schema: int = 1
    dim: int = 5120
    count: int = 0
    mode: str = "t2v"
    encoder: str = "unknown"
    source: str = ""
    created_at: str = ""
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def _pack(seqs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.asarray([s.shape[0] for s in seqs], dtype=np.int32)
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float16), lens
    flat = np.concatenate(seqs, axis=0)
    return flat, lens


def _unpack(flat: np.ndarray, lens: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    off = 0
    for n in lens:
        n = int(n)
        out.append(flat[off : off + n])
        off += n
    return out


class PairDataset:
    """内存中的成对数据集：``x`` = 负例（逻辑欠缺），``y`` = 正例（武打逻辑完备）。"""

    def __init__(self, meta: Optional[PairMeta] = None) -> None:
        self.meta = meta or PairMeta()
        self.x: List[np.ndarray] = []
        self.y: List[np.ndarray] = []
        self.x_score: List[float] = []
        self.y_score: List[float] = []
        self.records: List[Dict[str, Any]] = []
        # 不成对的样本：只给评分头当额外负例用（例如真实被筛掉的片段对应的
        # prompt）。桥训练需要"同一事件的正反对"，用不上这些，会跳过。
        self.extra: List[np.ndarray] = []
        self.extra_score: List[float] = []
        self.extra_records: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.x)

    def add(
        self,
        x_tokens: np.ndarray,
        y_tokens: np.ndarray,
        x_score: float = 0.0,
        y_score: float = 1.0,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        if x_tokens.ndim != 2 or y_tokens.ndim != 2:
            raise ValueError("token 序列必须是 [T, D]")
        if x_tokens.shape[-1] != y_tokens.shape[-1]:
            raise ValueError("正负例维度不一致，说明用了不同的文本编码器")
        self.x.append(np.asarray(x_tokens, dtype=np.float16))
        self.y.append(np.asarray(y_tokens, dtype=np.float16))
        self.x_score.append(float(x_score))
        self.y_score.append(float(y_score))
        self.records.append(record or {})
        self.meta.dim = int(x_tokens.shape[-1])
        self.meta.count = len(self.x)
        self.meta.records = self.records

    def add_unpaired(
        self,
        tokens: np.ndarray,
        score: float = 0.0,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        """加一条"没有配对"的样本，仅用于训练评分头（桥训练会忽略）。"""
        if tokens.ndim != 2:
            raise ValueError("token 序列必须是 [T, D]")
        self.extra.append(np.asarray(tokens, dtype=np.float16))
        self.extra_score.append(float(score))
        self.extra_records.append(record or {})
        self.meta.dim = int(tokens.shape[-1])
        if not self.x:
            self.meta.count = max(self.meta.count, len(self.extra))

    @property
    def pooled_x(self) -> np.ndarray:
        return np.stack([s.astype(np.float32).mean(axis=0) for s in self.x], axis=0)

    @property
    def pooled_y(self) -> np.ndarray:
        return np.stack([s.astype(np.float32).mean(axis=0) for s in self.y], axis=0)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        x_flat, x_len = _pack(self.x)
        y_flat, y_len = _pack(self.y)
        e_flat, e_len = _pack(self.extra)
        # 池化向量只是给诊断/快速统计用，用 fp16 存，省一半磁盘
        np.savez_compressed(
            path,
            x_flat=x_flat,
            x_len=x_len,
            y_flat=y_flat,
            y_len=y_len,
            e_flat=e_flat,
            e_len=e_len,
            x_pool=self.pooled_x.astype(np.float16) if len(self.x) else np.zeros((0, self.meta.dim), np.float16),
            y_pool=self.pooled_y.astype(np.float16) if len(self.y) else np.zeros((0, self.meta.dim), np.float16),
            x_score=np.asarray(self.x_score, dtype=np.float32),
            y_score=np.asarray(self.y_score, dtype=np.float32),
            e_score=np.asarray(self.extra_score, dtype=np.float32),
        )
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            f.write(self.meta.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "PairDataset":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到数据集：{path}")
        z = np.load(path)
        meta_path = os.path.splitext(path)[0] + ".json"
        meta = PairMeta()
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = PairMeta(**json.load(f))
        ds = cls(meta)
        xs = _unpack(z["x_flat"], z["x_len"])
        ys = _unpack(z["y_flat"], z["y_len"])
        x_score = z["x_score"].tolist() if "x_score" in z else [0.0] * len(xs)
        y_score = z["y_score"].tolist() if "y_score" in z else [1.0] * len(ys)
        for i, (a, b) in enumerate(zip(xs, ys)):
            ds.x.append(a)
            ds.y.append(b)
            ds.x_score.append(float(x_score[i]))
            ds.y_score.append(float(y_score[i]))
            ds.records.append(meta.records[i] if i < len(meta.records) else {})
        if "e_flat" in z and "e_len" in z:
            es = _unpack(z["e_flat"], z["e_len"])
            e_score = z["e_score"].tolist() if "e_score" in z else [0.0] * len(es)
            for i, a in enumerate(es):
                ds.extra.append(a)
                ds.extra_score.append(float(e_score[i]))
                ds.extra_records.append({"source": "unpaired"})
        ds.meta.count = len(ds.x)
        return ds

    def split(self, val_ratio: float = 0.1, seed: int = 1234) -> Tuple["PairDataset", "PairDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_val = max(1, int(round(len(self) * val_ratio))) if len(self) > 4 else 0
        val_idx = set(idx[:n_val].tolist())
        train, val = PairDataset(PairMeta(**{**self.meta.__dict__, "records": []})), PairDataset(
            PairMeta(**{**self.meta.__dict__, "records": []})
        )
        for i in range(len(self)):
            dst = val if i in val_idx else train
            dst.add(self.x[i], self.y[i], self.x_score[i], self.y_score[i], self.records[i])
        return train, val

    def stats(self) -> Dict[str, Any]:
        if not self.x:
            return {"count": 0}
        tx = [s.shape[0] for s in self.x]
        ty = [s.shape[0] for s in self.y]
        return {
            "count": len(self.x),
            "dim": self.meta.dim,
            "unpaired_extra": len(self.extra),
            "x_tokens": {"min": int(min(tx)), "max": int(max(tx)), "mean": float(np.mean(tx))},
            "y_tokens": {"min": int(min(ty)), "max": int(max(ty)), "mean": float(np.mean(ty))},
            "x_score_mean": float(np.mean(self.x_score)) if self.x_score else None,
            "y_score_mean": float(np.mean(self.y_score)) if self.y_score else None,
        }


def pool_tokens(flat: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """兼容旧格式：把紧凑序列池化成 [N, D]。"""
    pooled = []
    off = 0
    for n in lens:
        n = int(n)
        pooled.append(flat[off : off + n].astype(np.float32).mean(axis=0))
        off += n
    return np.stack(pooled, axis=0) if pooled else np.zeros((0, flat.shape[-1]), np.float32)
