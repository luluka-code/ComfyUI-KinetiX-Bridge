"""Combat Semantic Bridge Architecture and Weight I/O.

Only depends on torch, fully decoupled from ComfyUI for standalone testing and inference.

Supported Architectures:
-----------------------
``mlp``
    Isomorphic to community MiniMax-H3-Semantic-Bridge v1: per-token D -> H -> H -> D
    fully connected residual head. Provides direct compatibility with community 11MB weights.

``trans``
    Cross-token lightweight Transformer residual bridge. Sequence-level causality
    (weapon clashes, momentum recoil, distance consistency) requires inter-token modeling.
    Default: 2 layers / d=256 / 4 heads (~3.2M params, fp32 ≈ 13MB).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

SCHEMA_VERSION = 1
ARCH_MLP = "mlp"
ARCH_TRANS = "trans"
SUPPORTED_ARCHS = (ARCH_MLP, ARCH_TRANS)


@dataclass
class BridgeConfig:
    """Configuration and metadata for the combat semantic bridge."""

    arch: str = ARCH_TRANS
    dim: int = 5120
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    max_tokens: int = 4096
    mode: str = "t2v"           # t2v / ref2v / any
    schema: int = SCHEMA_VERSION
    name: str = "kinetix_bridge"
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(f"Unknown arch: {self.arch}, expected one of {SUPPORTED_ARCHS}")
        if self.arch == ARCH_MLP:
            if self.hidden <= 0:
                raise ValueError("hidden must be a positive integer")
        else:
            if self.hidden % self.heads != 0:
                raise ValueError(
                    f"hidden ({self.hidden}) must be divisible by heads ({self.heads})"
                )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _sinusoidal_positions(max_tokens: int, hidden: int) -> torch.Tensor:
    pos = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
    pe = torch.zeros(max_tokens, hidden, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.unsqueeze(0)                                   # [1, L, H]


class _ResidualMLP(nn.Module):
    """Token-wise MLP residual head (community isomorphic)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class _TokenTransformer(nn.Module):
    """Sequence-level cross-token lightweight Transformer."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden, bias=True)
        self.register_buffer(
            "_pe", _sinusoidal_positions(max_tokens, hidden), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.max_tokens = int(max_tokens)

    def _positions(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pe = self._pe
        if n <= pe.shape[1]:
            return pe[:, :n].to(device=device, dtype=dtype)
        # Super-long sequences (n > max_tokens): dynamically compute exact sinusoidal embeddings
        hidden = int(self.in_proj.out_features)
        return _sinusoidal_positions(n, hidden).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self._positions(h.shape[1], h.device, h.dtype)
        h = self.encoder(h)
        h = self.norm(h)
        return self.out_proj(h)


class WushuBridge(nn.Module):
    """Unified wrapper: forward(x) returns transformed embedding with the same shape."""

    def __init__(self, cfg: BridgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.arch == ARCH_MLP:
            self.net: nn.Module = _ResidualMLP(cfg.dim, cfg.hidden)
        else:
            self.net = _TokenTransformer(
                cfg.dim, cfg.hidden, cfg.layers, cfg.heads, cfg.dropout, cfg.max_tokens
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,D] input, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.cfg.dim:
            raise ValueError(
                f"Embedding dimension mismatch: model dim={self.cfg.dim}, input {x.shape[-1]}."
            )
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_bridge(cfg: BridgeConfig) -> WushuBridge:
    return WushuBridge(cfg)


def save_bridge(path: str, model: WushuBridge, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """Saves bridge to safetensors format with config metadata."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = {k: str(v) for k, v in asdict(model.cfg).items()}
    if extra_meta:
        for k, v in extra_meta.items():
            meta[f"train_{k}"] = str(v)
    tensors = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(tensors, path, metadata=meta)
    return path


def infer_config_from_state(state: Dict[str, torch.Tensor], meta: Dict[str, str]) -> BridgeConfig:
    """Infers architecture config from safetensors metadata and tensor shapes."""
    if "fc1.weight" in state:
        dim = int(state["fc1.weight"].shape[1])
        hidden = int(state["fc1.weight"].shape[0])
        return BridgeConfig(
            arch=ARCH_MLP,
            dim=dim,
            hidden=hidden,
            layers=2,
            heads=1,
            name=meta.get("name", "community_mlp_bridge"),
            note="Inferred from community MiniMax-H3-Semantic-Bridge tensor shapes",
        )

    if "net.fc1.weight" in state:
        dim = int(state["net.fc1.weight"].shape[1])
        hidden = int(state["net.fc1.weight"].shape[0])
        return BridgeConfig(
            arch=ARCH_MLP,
            dim=dim,
            hidden=hidden,
            layers=2,
            heads=1,
            name=meta.get("name", "kinetix_mlp_bridge"),
        )

    if "net.in_proj.weight" in state or "in_proj.weight" in state:
        key = "net.in_proj.weight" if "net.in_proj.weight" in state else "in_proj.weight"
        dim = int(state[key].shape[1])
        hidden = int(state[key].shape[0])
        prefix = "net.encoder.layers." if "net.in_proj.weight" in state else "encoder.layers."
        layer_ids = set()
        for k in state:
            if k.startswith(prefix):
                layer_ids.add(int(k[len(prefix) :].split(".")[0]))
        layers = max(layer_ids) + 1 if layer_ids else 2
        heads = int(meta.get("heads", 4)) if meta.get("heads") else 4
        return BridgeConfig(
            arch=ARCH_TRANS,
            dim=dim,
            hidden=hidden,
            layers=layers,
            heads=heads,
            name=meta.get("name", "kinetix_trans_bridge"),
        )

    raise ValueError("Unrecognized bridge weights: matches neither MLP nor Transformer architecture.")


def load_bridge(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> WushuBridge:
    """Loads combat residual bridge (.safetensors)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find bridge weight file: {path}")

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    state = load_file(path, device="cpu")
    cfg = infer_config_from_state(state, meta)

    model = WushuBridge(cfg)
    clean_state = {}
    for k, v in state.items():
        if not k.startswith("net.") and hasattr(model, "net"):
            clean_state[f"net.{k}"] = v
        else:
            clean_state[k] = v
    model.load_state_dict(clean_state, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
