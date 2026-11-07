# ComfyUI-KinetiX-Bridge
**MiniMax-H3 5120-d Combat Semantic Bridge (Runtime Inference)**

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/Model-MiniMax--H3-orange.svg)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)]()

专为 **MiniMax-H3** 动作视频打造的原生 5120 维物理因果与打击感语义残差桥（纯推理运行时）。
无需重新微调庞大的扩散底模，通过加载预训练的轻量级 `.safetensors` 语义桥模型，直接在 Conditioning 流形空间中修正动作因果，攻克 AI 打斗中常见的**“空气假打、各打各的、木偶人受击无硬直”**三大通病。

---

## 🌟 核心特性 (Key Features)

1. **5120 维原生流形切向注入**：
   - 严格在 MiniMax-H3 原生 5120 维语义空间内执行切向残差变换，保持模长与语义锚点（$1-\cos < 0.02$），**动作爆燃的同时人物面部、服装与场景风格 0 变异**。
2. **Ref2VA（图生视频/首尾帧）专属 `tail` 保护模式**：
   - 自动识别图像视觉 Token（通常 2000~5000+ 个 Token），开启 `token_span = tail` 即可**绝对冻结参考图视觉特征**，只精准微调尾部的打斗动作描述。
3. **原生支持超长序列（5412+ Tokens）**：
   - 采用解析正弦位置编码（Sinusoidal Position Embeddings），突破固定上下文长度限制，任意超长图生视频工作流永不报张量维度不匹配错误。
4. **极致轻量（Runtime Only）**：
   - 纯 Python 编写，零冗余训练依赖，即装即用。

---

## 📦 节点说明 (Node Overview)

在 ComfyUI 节点右键菜单的 **`MiniMax H3/KinetiX Bridge`** 分类下：

- **`KinetiX 武打语义逻辑桥 (Combat Semantic Bridge)`**：
  - 核心推理节点，串接在 `CLIPTextEncode` 与采样器之间，将预训练的打斗因果残差注入 Conditioning。
- **`KinetiX 清理缓存 (Clear Cache)`**：
  - 辅助工具节点，一键释放驻留显存的桥模型。

---

## 🚀 安装与使用 (Installation & Quick Start)

### 1. 安装插件

在你的 ComfyUI 根目录下的 `custom_nodes` 执行克隆：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/luluka-code/ComfyUI-KinetiX-Bridge.git
```

### 2. 模型存放

将训练好或下载的 `.safetensors` 语义桥模型放置在：
```text
ComfyUI/models/wushu_bridge/
```
*(例如：`kinetix_combat_bridge_v1.safetensors`)*

---

## 🎛️ 黄金推荐参数指南 (Recommended Settings)

在 **`KinetiX 武打语义逻辑桥`** 节点上，请根据生成场景选择推荐配置：

### 模式 A：文生视频 (T2V) —— 极致击飞与硬碰硬
- **`bridge`**：选择你的权重（如 `kinetix_combat_bridge_v1.safetensors`）
- **`alpha`**：`0.18 ~ 0.25`（获得极具张力的卡肉、碰撞火星与受击退步）
- **`magnitude_match`**：`per_token`（保护特征模长守恒，防过曝偏色）
- **`token_span`**：`all`

### 模式 B：图生视频 / 首尾帧 (Ref2VA) —— 保护人脸长相
- **`bridge`**：选择你的权重
- **`alpha`**：`0.10 ~ 0.12`（温和释放动作）
- **`magnitude_match`**：`per_token`
- **`token_span`**：`tail` ⭐️ **（必选！跳过参考图图像 Token，防人脸走样）**
- **`tail_ratio`**：`0.5`

---

## 📄 License
本项目基于 Apache-2.0 开源协议。
