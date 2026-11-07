"""Applies residual combat bridge to ComfyUI CONDITIONING.

ComfyUI CONDITIONING structure::

    [[embedding: Tensor[B, T, D], metadata: dict], ...]        # typically 1 element

MiniMax-H3 has D = 5120. Metadata may contain ``minimax_keyframes``,
``minimax_frame_count``, ``minimax_refs``, etc. (first/last frames, reference images/audios).
This module only transforms the embedding tensor, preserving every key in metadata intact,
which guarantees seamless compatibility with multi-reference and I2V pipelines.

Transformation formula::

    x_norm    = rms_normalize(h)
    pred      = bridge(x_norm)
    pred      = magnitude_match(pred, h)        # per_token / global / none
    hybrid    = h + alpha * (pred - h)

Low ``alpha`` (0.08~0.25) provides gentle directional guidance towards physically complete combat;
alpha=1 fully replaces the embedding and is not recommended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .bridge_model import WushuBridge

MAGNITUDE_MODES = ("per_token", "global", "none")
TEXT_SPAN_MODES = ("all", "tail")

META_KEY_PREFIX = "kinetix_bridge_"


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x / rms


def magnitude_match(source: torch.Tensor, target: torch.Tensor, mode: str = "per_token") -> torch.Tensor:
    """Aligns energy magnitude of source to target to avoid overall brightness/color shifts."""
    source = source.float()
    target = target.float()
    if mode == "none":
        return source
    if mode == "per_token":
        s = torch.sqrt(source.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        t = torch.sqrt(target.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        return source * (t / s)
    if mode == "global":
        s = torch.sqrt(source.pow(2).mean() + 1e-8)
        t = torch.sqrt(target.pow(2).mean() + 1e-8)
        return source * (t / s)
    raise ValueError(f"Unknown magnitude_match mode: {mode}, expected one of {MAGNITUDE_MODES}")


@dataclass
class ApplyReport:
    """Observable telemetry returned directly to ComfyUI STRING output."""

    applied: int = 0
    skipped: int = 0
    dims: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    drift: List[float] = field(default_factory=list)      # 1-cos relative to original embedding
    ref_like: List[bool] = field(default_factory=list)     # whether conditioning contains image/keyframe tokens
    messages: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"KinetiX Combat Bridge: Applied {self.applied} conditioning segment(s), skipped {self.skipped} segment(s)",
        ]
        for i, (d, t, dr) in enumerate(zip(self.dims, self.tokens, self.drift)):
            flag = " (Contains Ref/Keyframe tokens)" if i < len(self.ref_like) and self.ref_like[i] else ""
            lines.append(f"  #{i}: dim={d} tokens={t} semantic_drift(1-cos)={dr:.4f}{flag}")
        lines.extend(self.messages)
        return "\n".join(lines)


def _looks_reference_conditioned(metadata: Dict[str, Any]) -> bool:
    for key in ("minimax_refs", "minimax_frame_count", "minimax_keyframes"):
        if metadata.get(key):
            return True
    return False


def _token_slice(tokens: int, span: str, tail_ratio: float) -> Tuple[int, int]:
    """Selects the token range to apply the bridge.

    In H3 packaged sequences, reference image/audio tokens and text tokens are packed together.
    'all' transforms the full sequence; 'tail' only modifies the trailing tail_ratio proportion,
    which acts as a safe guard to preserve reference character face identity.
    """
    if span == "all" or tail_ratio >= 1.0:
        return 0, tokens
    start = max(0, int(round(tokens * (1.0 - max(0.0, tail_ratio)))))
    return start, tokens


def _apply_chunked(
    bridge: WushuBridge,
    x: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    if chunk_tokens <= 0 or x.shape[1] <= chunk_tokens:
        return bridge(x)
    chunks = []
    for s in range(0, x.shape[1], chunk_tokens):
        part = x[:, s : s + chunk_tokens]
        chunks.append(bridge(part))
    return torch.cat(chunks, dim=1)


def apply_bridge_to_conditioning(
    conditioning: Sequence[Any],
    bridge: WushuBridge,
    alpha: float = 0.18,
    magnitude_match_mode: str = "per_token",
    token_span: str = "all",
    tail_ratio: float = 0.5,
    max_tokens_per_chunk: int = 0,
    allow_dim_mismatch: bool = False,
) -> Tuple[List[Any], ApplyReport]:
    """Applies residual combat bridge to conditioning segments, returning (new_conditioning, report)."""
    if magnitude_match_mode not in MAGNITUDE_MODES:
        raise ValueError(f"magnitude_match must be one of {MAGNITUDE_MODES}")
    if token_span not in TEXT_SPAN_MODES:
        raise ValueError(f"token_span must be one of {TEXT_SPAN_MODES}")

    report = ApplyReport()
    new_cond: List[Any] = []
    device = next(bridge.parameters()).device

    for idx, item in enumerate(conditioning):
        if not (isinstance(item, (list, tuple)) and len(item) >= 2):
            raise RuntimeError("Invalid CONDITIONING structure: expected [[tensor, metadata], ...]")

        native = item[0]
        meta = dict(item[1]) if isinstance(item[1], dict) else {}

        if not torch.is_tensor(native) or native.ndim != 3:
            report.skipped += 1
            report.messages.append(
                f"  [!] #{idx} Skipped: expected [B,T,D] tensor, got {type(native)}"
            )
            new_cond.append(item)
            continue

        if native.shape[-1] != bridge.cfg.dim:
            msg = f"#{idx} Dimension mismatch: conditioning dim={native.shape[-1]}, bridge dim={bridge.cfg.dim}"
            if not allow_dim_mismatch:
                raise ValueError(
                    msg + ". Please use the matching H3 text encoder or bridge weights."
                )
            report.skipped += 1
            report.messages.append("  [!] " + msg + " (skipped)")
            new_cond.append(item)
            continue

        b, t, d = native.shape
        is_ref = _looks_reference_conditioned(meta)
        start, end = _token_slice(t, token_span, tail_ratio)

        # Move to bridge device to avoid cross-device tensor mismatch
        h = native[:, start:end, :].to(device=device, dtype=torch.float32)

        if h.shape[1] == 0:
            report.skipped += 1
            report.messages.append(f"  [!] #{idx} Selected token slice is empty ({start}:{end}), skipped")
            new_cond.append(item)
            continue

        x_norm = rms_normalize(h)

        with torch.no_grad():
            pred = _apply_chunked(bridge, x_norm, max_tokens_per_chunk)
            pred = magnitude_match(pred, h, mode=magnitude_match_mode)
            hybrid = h + float(alpha) * (pred - h)

            # Compute semantic drift (cosine distance)
            p_orig = h.mean(dim=1)
            p_hyb = hybrid.mean(dim=1)
            sim = torch.nn.functional.cosine_similarity(p_orig, p_hyb, dim=-1).clamp(-1.0, 1.0)
            drift_val = float((1.0 - sim).mean().item())

        out_tensor = native.clone()
        out_tensor[:, start:end, :] = hybrid.to(device=native.device, dtype=native.dtype)

        meta[f"{META_KEY_PREFIX}applied"] = True
        meta[f"{META_KEY_PREFIX}alpha"] = float(alpha)
        meta[f"{META_KEY_PREFIX}drift"] = drift_val

        new_cond.append([out_tensor, meta])
        report.applied += 1
        report.dims.append(int(d))
        report.tokens.append(int(t))
        report.drift.append(drift_val)
        report.ref_like.append(is_ref)

    if any(report.ref_like) and token_span != "tail":
        report.messages.append(
            "  [i] Notice: Conditioning contains reference image / keyframe tokens. "
            "For optimal face identity preservation, setting token_span='tail' with alpha <= 0.12 is strongly recommended."
        )

    return new_cond, report
