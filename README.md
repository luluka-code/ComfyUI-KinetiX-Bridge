# ComfyUI-KinetiX-Bridge
**MiniMax-H3 5120-d Combat Semantic Bridge (Pure Inference Runtime)**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)]()
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/kinetixfx/Kinetix_Combat_Bridge)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)]()

A lightweight runtime node for **MiniMax-H3** that injects authentic physical combat causality, heavy impact recoil, and contact alignment into action video generation without facial deformation or style drift.

---

## 🚀 Quick Start

### 1. Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/luluka-code/ComfyUI-KinetiX-Bridge.git
```
Restart ComfyUI after installation.

### 2. Model Loading (Auto Download or Manual)

* **Option A: Auto Download (Recommended)**  
  Simply leave the `bridge` dropdown on:  
  `kinetix_combat_bridge_v1.safetensors (Auto Download)`  
  The node will automatically download the official weights (~16 MB) from [Hugging Face (kinetixfx/Kinetix_Combat_Bridge)](https://huggingface.co/kinetixfx/Kinetix_Combat_Bridge) directly into `ComfyUI/models/kinetix_bridge/`. Built-in mirror fallback is included.

* **Option B: Manual Download**  
  Download [`kinetix_combat_bridge_v1.safetensors`](https://huggingface.co/kinetixfx/Kinetix_Combat_Bridge/blob/main/kinetix_combat_bridge_v1.safetensors) manually and place it into any of the scanned directories:
  * `ComfyUI/models/kinetix_bridge/` *(Recommended standard)*
  * `ComfyUI/models/loras/`
  * `ComfyUI/models/wushu_bridge/`
  * `ComfyUI/custom_nodes/ComfyUI-KinetiX-Bridge/models/`

---

## 🎛️ Recommended Presets

### Mode A: Text-to-Video (T2V) — Maximum Action Impact
* **`alpha`**: `0.18 ~ 0.25` (intense sparks, blade clashes, and staggered knockbacks)
* **`magnitude_match`**: `per_token` (prevents color shifts)
* **`token_span`**: `all`

### Mode B: Reference-to-Video (Ref2VA) — Face Identity Guard
* **`alpha`**: `0.10 ~ 0.12` (smooth combat flow)
* **`magnitude_match`**: `per_token`
* **`token_span`**: `tail` ⭐️ *(Freezes reference image tokens to protect face & costume identity)*
* **`tail_ratio`**: `0.5`

---

## 📦 Node Menu
Found under: **`MiniMax H3/KinetiX Bridge`** -> `KinetiX Combat Semantic Bridge`

## 📄 License
Apache-2.0
