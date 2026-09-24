"""
评估指标汇总脚本
功能：整理 Ultralytics 训练结果，输出指标表到 docs/实验记录.md
"""
import json
import csv
from pathlib import Path
from datetime import datetime


def parse_results(results_dir):
    """
    从 Ultralytics runs/ 目录解析指标
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"[警告] 未找到结果文件: {csv_path}")
        return None

    # 读取最后一行作为最终指标
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    last = rows[-1]
    # Ultralytics CSV 列名示例：
    # epoch, time, train/box_loss, train/cls_loss, ..., metrics/mAP50(B), metrics/mAP50-95(B)
    metrics = {
        "epochs": len(rows),
        "mAP50": float(last.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(last.get("metrics/mAP50-95(B)", 0)),
        "precision": float(last.get("metrics/precision(B)", 0)),
        "recall": float(last.get("metrics/recall(B)", 0)),
    }
    return metrics


def generate_report(experiments, out_md="docs/实验记录.md"):
    """
    生成 Markdown 指标表
    experiments: list of dict {name, model, imgsz, results_dir}
    """
    lines = ["# 实验记录\n", f"> 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    lines.append("| 实验 | 模型 | imgsz | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Epochs |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for exp in experiments:
        metrics = parse_results(exp["results_dir"])
        if metrics:
            line = (
                f"| {exp['name']} | {exp['model']} | {exp['imgsz']} | "
                f"{metrics['mAP50']:.4f} | {metrics['mAP50_95']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['epochs']} |"
            )
        else:
            line = f"| {exp['name']} | {exp['model']} | {exp['imgsz']} | - | - | - | - | - |"
        lines.append(line)

    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"实验记录已更新: {out_path.absolute()}")


if __name__ == "__main__":
    # 示例：手动填写实验列表后运行
    experiments = [
        {
            "name": "baseline_n",
            "model": "YOLO11n",
            "imgsz": 640,
            "results_dir": "runs/baseline_n/exp1",
        },
    ]
    generate_report(experiments)
