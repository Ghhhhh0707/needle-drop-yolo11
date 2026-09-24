# 电脑横机织布漏针（破洞/脱线）智能检测 — YOLO11

> **项目阶段**：阶段0 仓库与环境搭建（进行中）  
> **计划周期**：16周 ｜ **技术栈**：YOLO11 + Python

---

## 问题背景

电脑横机（针织机）在织造过程中，因纱线断头、织针跳针等原因会产生**漏针**，在布面上表现为**破洞 / 脱线区域**（纱线松散卷曲外翻、纹理断口）。次品流出后才被人工发现，成本高、漏检多。

本项目目标：在横机出料口照灯前加装摄像头，用目标检测模型实时识别漏针，替代人工目检。

---

## 我的贡献（进行中）

- [ ] 自采工业数据集（约 500 张，五色覆盖，采集规范见 [docs/数据采集规范.md](docs/数据采集规范.md)）
- [ ] 完整标注体系：微信画圈 → YOLO 坐标导出 → 7:2:1 按批次分层
- [ ] 训练闭环：YOLO11 基线 → 改进 → 消融（指标表见 [docs/实验记录.md](docs/实验记录.md)）
- [ ] 部署：ONNX/OpenVINO 实时推理 + Gradio Demo

---

## 快速开始

### 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/needle-drop-yolo11.git
cd needle-drop-yolo11

# 2. 创建环境
conda create -n needle python=3.10 -y
conda activate needle
pip install -r requirements.txt

# 3. 验证
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 训练（阶段3）

```bash
# 基线训练
yolo detect train data=data/data.yaml model=yolo11n.pt epochs=150 imgsz=640 batch=16
```

### 部署（阶段5）

```bash
# 导出 ONNX
python deploy/export_onnx.py --weights runs/baseline_n/exp1/weights/best.pt --format onnx

# 启动 Demo
python deploy/gradio_app.py
```

---

## 项目结构

```
needle-drop-yolo11/
├── data/               # 数据集（图片和标注，raw/不进Git）
├── scripts/            # 数据处理和评估脚本
├── configs/            # 训练配置
├── deploy/             # 导出和推理代码
├── docs/               # 规范和实验记录
├── examples/           # 样例结果图
└── README.md
```

---

## 数据声明

因项目协议，**完整数据集不公开**。本仓库仅发布代码、少量脱敏样例图及复现方法。学术引用或合作请联系项目组。

---

## 引用

```
[1] JOCHER G, QIU J. Ultralytics YOLO11: version 11.0.0[CP/OL]. 2024. https://github.com/ultralytics/ultralytics.
[2] KHANAM R, HUSSAIN M. YOLOv11: an overview of the key architectural enhancements[EB/OL]. arXiv:2410.17725, 2024.
```

## License

本项目代码采用 [MIT License](LICENSE)。
