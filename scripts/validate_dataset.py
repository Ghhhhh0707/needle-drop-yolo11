"""
数据集校验脚本
检查项：
1. 每张图片有同名TXT，反之亦然（无孤儿文件）
2. 所有坐标在 [0,1] 范围内
3. 框宽高 > 0
4. 类别统计
"""
import os
import sys
from pathlib import Path
from collections import Counter


def validate_split(split_dir, split_name):
    """
    校验一个划分（train/val/test）
    """
    img_dir = Path(split_dir) / "images" / split_name
    lbl_dir = Path(split_dir) / "labels" / split_name

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[跳过] 目录不存在: {img_dir} 或 {lbl_dir}")
        return False

    images = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

    errors = []
    warnings = []
    class_counter = Counter()
    total_boxes = 0

    # 孤儿文件检查
    img_only = set(images.keys()) - set(labels.keys())
    lbl_only = set(labels.keys()) - set(images.keys())

    if img_only:
        warnings.append(f"无标注的图片: {len(img_only)} 张")
        for name in sorted(img_only)[:5]:
            warnings.append(f"  - {name}")
    if lbl_only:
        warnings.append(f"无图片的标注: {len(lbl_only)} 个")
        for name in sorted(lbl_only)[:5]:
            warnings.append(f"  - {name}")

    # 校验每个标注文件
    for stem, lbl_path in labels.items():
        if stem not in images:
            continue

        img_path = images[stem]
        # 读取图片尺寸
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            errors.append(f"无法读取图片 {img_path}: {e}")
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{lbl_path.name} 第{line_num}行: 格式错误 ({len(parts)} 列)")
                continue

            try:
                cls, xc, yc, w, h = map(float, parts)
                cls = int(cls)
            except ValueError:
                errors.append(f"{lbl_path.name} 第{line_num}行: 数值解析失败")
                continue

            total_boxes += 1
            class_counter[cls] += 1

            # 范围检查
            for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"{lbl_path.name} 第{line_num}行: {name}={val} 越界 [0,1]")

            if w <= 0 or h <= 0:
                errors.append(f"{lbl_path.name} 第{line_num}行: 框宽高为零或负 (w={w}, h={h})")

    # 输出报告
    print(f"\n{'='*50}")
    print(f"校验报告: {split_name}")
    print(f"{'='*50}")
    print(f"图片数: {len(images)}, 标注数: {len(labels)}")
    print(f"总检测框: {total_boxes}")
    print(f"类别分布: {dict(class_counter)}")

    if warnings:
        print(f"\n警告 ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print(f"\n错误 ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return False
    else:
        print("  ✓ 校验通过")
        return True


def validate_all(data_dir="data"):
    data_dir = Path(data_dir)
    all_pass = True
    for split in ["train", "val", "test"]:
        ok = validate_split(data_dir, split)
        all_pass = all_pass and ok

    print(f"\n{'='*50}")
    if all_pass:
        print("✓ 全部校验通过")
    else:
        print("✗ 存在错误，请修复后重新运行")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--data", type=str, default="data", help="Data directory")
    args = parser.parse_args()
    validate_all(args.data)
