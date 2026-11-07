# ComfyUI-KinetiX-Bridge
**MiniMax-H3 5120-d Combat Semantic Bridge & High-Impact Action Engine**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)]()

专为 **MiniMax-H3** 动作视频打造的原生 5120 维物理因果与打击感语义残差桥。在无需重训庞大扩散底模的前提下，通过对文本与参考图 Conditioning 空间的流形切空间定向微扰，彻底攻克 AI 打斗中常见的**“空气假打、各打各的、木偶人受击无硬直”**三大通病。

---

## 🌟 核心特性

1. **5120 维原生流形切向注入**：
   - 严格在 MiniMax-H3 文本编码器原生的 5120 维潜空间内操作，保持模长与语义锚点（$1-\cos < 0.02$），**动作爆燃的同时人物面部、服装与画风 0 变异**。
2. **Ref2VA（图生视频/首尾帧）专属 `tail` 保护模式**：
   - 自动识别图像视觉 Token（通常 2000~5000+ 个 Token），开启 `token_span = tail` 即可**绝对冻结参考图人脸特征**，只精准微调尾部打斗动作描述。
3. **原生支持超长序列（5412+ Tokens）**：
   - 采用解析正弦位置编码（Sinusoidal Position Embeddings），突破固定上下文长度限制，任意超长图生视频工作流永不报张量维度不匹配。
4. **内置 27 类硬核动作与物理降级算子**：
   - 支持从动作正例文本自动合成力线缺失、受击木偶化、法术射空等成对训练样本，支持一键脱机或在 ComfyUI 内原位训练。

---

## 📦 节点总览 (Core Nodes)

| 节点名称 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`KinetiX 武打语义逻辑桥`**<br>`(KinetiXSemanticBridge)` | 推理端核心 | 串接在 CLIPTextEncode 与采样器之间，将训练好的权重注入 Conditioning。 |
| **`KinetiX 构建训练对`**<br>`(KinetixBuildPairs)` | 数据端工具 | 读取正例动作提示词，用 27 种物理算子自动合成负例并编码成 5120 维 `.npz` 数据集。 |
| **`KinetiX 训练残差桥`**<br>`(KinetixTrainBridge)` | 训练端核心 | 1 键原位训练轻量级 5120 维 Transformer 残差桥模型（`.safetensors`）。 |
| **`KinetiX 清理缓存`**<br>`(KinetixClearCache)` | 辅助工具 | 释放驻留显存的桥模型与中间张量缓存。 |

---

## 🚀 快速上手 (Quick Start)

### 1. 安装 (Installation)

在你的 ComfyUI 根目录下的 `custom_nodes` 执行克隆：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/luluka-code/ComfyUI-KinetiX-Bridge.git
```

重启 ComfyUI 即可自动加载节点。

### 2. 模型存放目录

训练或下载的 `.safetensors` 语义桥模型请放置在以下任一目录：
- `ComfyUI/models/wushu_bridge/`（推荐）
- `ComfyUI/custom_nodes/ComfyUI-KinetiX-Bridge/models/`

---

## 🎛️ 黄金推荐参数指南

在 **`KinetiX 武打语义逻辑桥`** 节点上，请根据生成场景选择推荐配置：

### 模式 A：文生视频 (T2V) —— 极致击飞与硬碰硬
- **`bridge`**：选择你的训练权重（如 `kinetix_combat_bridge_v1.safetensors`）
- **`alpha`**：`0.18 ~ 0.25`（获得极具张力的卡肉、碰撞火星与受击退步）
- **`magnitude_match`**：`per_token`（保护模长绝对守恒，防过曝偏色）
- **`token_span`**：`all`
- **`judge`**：`none`
- **`guard`**：`False`

### 模式 B：图生视频 / 首尾帧 (Ref2VA) —— 保护人脸长相
- **`bridge`**：选择你的训练权重
- **`alpha`**：`0.10 ~ 0.12`（温和释放动作）
- **`magnitude_match`**：`per_token`
- **`token_span`**：`tail` ⭐️ **（必选！跳过参考图图像 Token，防人脸走样）**
- **`tail_ratio`**：`0.5`
- **`judge`**：`none`
- **`guard`**：`False`

---

## 🏋️‍♂️ 训练属于自己的语义桥 (Training)

1. 打开工作流 `workflows/01_Train_KinetiX_Bridge.json`；
2. 连接 MiniMax-H3 的 Text Encoder（CLIP）；
3. 语料路径填入包含动作正例的 `.jsonl`（本项目 `datasets/kinetix_combat_pairs_all.jsonl` 已内置 100+ 条院线级实证动作语料）；
4. 点击 `Queue Prompt`，系统将自动编码数据并执行 60 轮自适应早停训练，约 1~2 分钟即可生成你自己的专属 `.safetensors`！

---

## 📄 License
本项目基于 Apache-2.0 开源协议。欢迎动作导演、视效从业者与二次开发者共同维护！
