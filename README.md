# 🚑 Ambulense: The First-Aid Kit for VLA Trajectories

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Ambulense** 是一款专为具身智能（Embodied AI）和灵巧手（Dexterous Hand）研究人员打造的轻量级 Web 数据“急救”与审查工具。

在训练 Vision-Language-Action (VLA) 模型时，专家演示数据中常夹杂着抖动、丢包或失误操作——这些“生病”的数据会严重污染模型策略。Ambulense 就像数据清洗的“救护车”，让你能够以极低成本**同步回放摄像头视角与多自由度关节运动**，快速诊断异常帧，并“抢救”出高质量的轨迹数据。

---

## ✨ 核心特性 (Key Features)

- ⏱️ **多模态时间轴完美同步**：拖动进度条，摄像头画面与所有关节的运动学曲线严格对齐。
- 🔍 **多自由度专注模式**：灵巧手自由度太多？支持一键勾选/屏蔽特定关节（Joints），让“诊断报告”告别杂乱。
- ⚠️ **异常抖动自动预警**：内置阈值检测算法，自动在发生关节突变（如传感器丢包或碰撞抖动）的帧抛出视觉警告。
- 🏷️ **极简分诊与导出**：一键对当前 Episode 进行“分诊”标记（🟢 优秀 Keep / 🟡 瑕疵 Needs Trim / 🔴 废弃 Discard），自动导出诊断日志 `clean_dataset_meta.json`。
- 🔌 **零配置数据接入**：只需将采集的 `.npz` 轨迹数据扔进 `data/` 文件夹，系统会自动解析轨迹长度与关节维度。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境安装
克隆本仓库并安装基础依赖（推荐使用虚拟环境）：
```bash
git clone [https://github.com/YourUsername/Ambulense.git](https://github.com/YourUsername/Ambulense.git)
cd Ambulense
pip install streamlit numpy pandas pillow altair
```


### 2. 生成测试数据
无需自己先辛苦采数据，运行内置的生成脚本，快速体验“急救”流程：

```bash
python generate_samples.py：
```
(这将在 data/ 目录下生成三个包含人工制造异常的 .npz 测试文件。)

### 3. 启动 Ambulense
```bash
streamlit run app.py
```
浏览器将自动打开网页端。

## 📂 如何接入真实数据进行“诊断”？
Ambulense 采用高度解耦的模板化设计。你只需要将采集好的单个 Episode 存为通用的 .npz 压缩包，放入 data/ 文件夹即可。

数据格式要求：
单个 .npz 文件需要包含以下两个 Key：

images: 摄像头图像序列。形状必须为 (T, H, W, 3)，类型为 uint8。

joint_angles: 机器人/灵巧手的关节角度序列。形状必须为 (T, N)，其中 T 是总帧数，N 是关节总自由度。

(注：如果你使用的是 HDF5 或 ROS Bag 格式，只需在 app.py 中的 load_real_episode 函数内修改两行读取逻辑即可无缝接入。)
