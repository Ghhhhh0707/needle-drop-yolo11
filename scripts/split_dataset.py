"""
数据集划分脚本
功能：按批次分层，7:2:1 划分 train/val/test（防止同批次泄漏）
"""
import os
import shutil
import random
import csv
from pathlib import Path
from collections import defaultdict


def extract_batch(filename):
    """
    从文件名提取批次号
    支持格式：微信图片_20260920142725_188_28.jpg -> 批次号 28
    """
    name = Path(filename).stem
    parts = name.split("_")
    # 尝试取最后一段数字作为批次
    for part in reversed(parts):
        if part.isdigit():
            return part
    return "unknown"


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    按批次分层划分数据集
    """
    random.seed(seed)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # 收集图片并按批次分组
    batch_groups = defaultdict(list)
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in image_dir.glob(ext):
            batch = extract_batch(img_path.name)
            batch_groups[batch].append(img_path)

    batches = list(batch_groups.keys())
    random.shuffle(batches)

    n = len(batches)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_batches = set(batches[:n_train])
    val_batches = set(batches[n_train:n_train + n_val])
    test_batches = set(batches[n_train + n_val:])

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for batch, files in batch_groups.items():
        if batch in train_batches:
            splits["train"].extend(files)
        elif batch in val_batches:
            splits["val"].extend(files)
        else:
            splits["test"].extend(files)

    # 创建输出目录并复制文件
    for split_name, file_list in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in file_list:
            # 复制图片
            shutil.copy2(img_path, img_out / img_path.name)
            # 复制对应标注
            txt_path = label_dir / (img_path.stem + ".txt")
            if txt_path.exists():
                shutil.copy2(txt_path, lbl_out / txt_path.name)
            else:
                print(f"[警告] 无标注文件: {txt_path}")

    # 写划分清单CSV
    csv_path = output_dir / "split_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "batch", "split"])
        for split_name, file_list in splits.items():
            for img_path in file_list:
                batch = extract_batch(img_path.name)
                writer.writerow([img_path.name, batch, split_name])

    # 统计
    print("=" * 50)
    print("数据集划分完成")
    print("=" * 50)
    for split_name in ["train", "val", "test"]:
        count = len(splits[split_name])
        pct = count / sum(len(v) for v in splits.values()) * 100
        print(f"  {split_name:5s}: {count:4d} 张 ({pct:.1f}%)")
    print(f"\n划分清单: {csv_path}")
    print(f"批次统计: 共 {len(batches)} 个批次")
    print(f"  train batches: {len(train_batches)}")
    print(f"  val   batches: {len(val_batches)}")
    print(f"  test  batches: {len(test_batches)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset by batch (7:2:1)")
    parser.add_argument("--images", type=str, default="data/raw", help="Source image directory")
    parser.add_argument("--labels", type=str, default="data/labels/all", help="Source label directory")
    parser.add_argument("--out", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(args.images, args.labels, args.out, seed=args.seed)
