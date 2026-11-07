# ComfyUI-KinetiX-Bridge
**MiniMax-H3 5120-d Combat Semantic Bridge (Pure Inference Runtime)**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)]()

A high-performance, native 5120-d physical causality and combat semantic bridge designed specifically for **MiniMax-H3** action video generation.
Without fine-tuning the heavy DiT diffusion base model, it directly operates in the 5120-d Conditioning manifold to solve common AI combat issues such as **"phantom punching, disconnected actions, and mannequin-like lack of hit reactions"**.

---

## 🌟 Key Features

1. **5120-d Native Manifold Tangent Projection**:
   - Operates strictly within the native 5120-d latent space of the MiniMax-H3 text encoder. Maintains energy magnitude and semantic anchors ($1 - \cos < 0.02$), delivering intense combat impact while **strictly preserving character facial identity, costume details, and scene artstyle**.
2. **Ref2VA `tail` Guard Mode (Image-to-Video / Keyframe Protection)**:
   - Automatically detects visual tokens in reference image conditioning (typically 2,000 to 5,000+ tokens). Enabling `token_span = tail` **absolutely freezes reference face embeddings**, modifying only the trailing action tokens.
3. **Native Support for Super-Long Sequences (5412+ Tokens)**:
   - Employs analytical sinusoidal position embeddings, breaking fixed sequence length boundaries. Never throws tensor dimension mismatch errors on multi-reference long conditioning.
4. **Ultra-Lightweight Pure Runtime (< 30 KB)**:
   - Zero training dependencies, instant cold boot, and minimal VRAM footprint.

---

## 📦 Node Overview

Located under **`MiniMax H3/KinetiX Bridge`** in the ComfyUI node menu:

- **`KinetiX Combat Semantic Bridge`** (`KinetiXSemanticBridge`):
  - Injects pre-trained combat physical causality into Conditioning between `CLIPTextEncode` and the sampler.
- **`KinetiX Clear Cache`** (`KinetiXClearCache`):
  - Frees cached bridge weights and clears intermediate VRAM allocations.

---

## 🚀 Installation & Quick Start

### 1. Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/luluka-code/ComfyUI-KinetiX-Bridge.git
```

Restart ComfyUI to load the nodes.

### 2. Model Weights Placement

Place your pre-trained `.safetensors` bridge weights into:
```text
ComfyUI/models/wushu_bridge/
```
*(Example: `kinetix_combat_bridge_v1.safetensors`)*

---

## 🎛️ Recommended Parameter Guide

### Mode A: Text-to-Video (T2V) — Maximum Kinetic Impact
- **`bridge`**: Select your `.safetensors` weight
- **`alpha`**: `0.18 ~ 0.25` (unleashes clashing sparks, heavy recoil, and staggered knockbacks)
- **`magnitude_match`**: `per_token` (preserves token energy to prevent saturation/color shift)
- **`token_span`**: `all`

### Mode B: Reference-to-Video / Keyframes (Ref2VA) — Face Identity Preservation
- **`bridge`**: Select your `.safetensors` weight
- **`alpha`**: `0.10 ~ 0.12` (smooth combat motion release)
- **`magnitude_match`**: `per_token`
- **`token_span`**: `tail` ⭐️ **(Crucial! Skips reference image visual tokens to prevent facial deformation)**
- **`tail_ratio`**: `0.5`

---

## 💡 Technical Insights

### Why 5120-d Semantic Bridge instead of LoRA?
MiniMax-H3's DiT text cross-attention layers are highly sensitive to prompt semantics. Traditional combat prompts frequently suffer from semantic attenuation (e.g., "dodges blade" turning into static staring). 

The KinetiX Bridge projects the prompt conditioning along the physical causality gradient derived from hundreds of film-grade action choreographies, enforcing:
- **Contact Alignment**: Blades and shields physically collide on the hit frame with recoil sparks.
- **Momentum Transfer**: The defender's center of mass staggers backward upon heavy impact.
- **Mutual Eye Gaze**: Opponents maintain locked line-of-sight during exchanges.

---

## 📄 License
This project is licensed under the Apache-2.0 License.
