"""
数据集校验脚本
检查：图片-标注对应、越界框、负值、空标注、类别ID有效性
"""
import argparse
from pathlib import Path
from PIL import Image


def validate_dataset(data_root):
    data_root = Path(data_root)
    issues = []
    stats = {"total_images": 0, "total_labels": 0, "total_boxes": 0, "empty_labels": 0}

    for split in ["train", "val", "test"]:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

        if not img_dir.exists():
            continue

        for img_f in sorted(img_dir.glob("*")):
            if img_f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            stats["total_images"] += 1
            lbl_f = lbl_dir / f"{img_f.stem}.txt"

            # Check label existence
            if not lbl_f.exists():
                issues.append(f"[MISSING] {split}/{img_f.name}: no matching label")
                continue

            stats["total_labels"] += 1

            # Load image size
            try:
                with Image.open(img_f) as img:
                    img_w, img_h = img.size
            except Exception as e:
                issues.append(f"[CORRUPT] {split}/{img_f.name}: {e}")
                continue

            # Validate annotations
            with open(lbl_f) as f:
                lines = f.readlines()

            if not lines or all(not l.strip() for l in lines):
                stats["empty_labels"] += 1
                issues.append(f"[EMPTY] {split}/{lbl_f.name}: no annotations")
                continue

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"[FORMAT] {split}/{lbl_f.name} line {line_num}: expected 5 values, got {len(parts)}")
                    continue

                try:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    issues.append(f"[FORMAT] {split}/{lbl_f.name} line {line_num}: non-numeric values")
                    continue

                if cls_id != 0:
                    issues.append(f"[CLASS] {split}/{lbl_f.name} line {line_num}: class_id={cls_id}, expected 0")

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    issues.append(f"[BOUNDS] {split}/{lbl_f.name} line {line_num}: normalized values out of [0,1]")

                if w <= 0 or h <= 0:
                    issues.append(f"[SIZE] {split}/{lbl_f.name} line {line_num}: width={w}, height={h}")

                # Check if box extends beyond image (strict check)
                if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                    issues.append(f"[OVERFLOW] {split}/{lbl_f.name} line {line_num}: box exceeds image bounds")

                stats["total_boxes"] += 1

    # Print results
    print("=" * 50)
    print("Dataset Validation Report")
    print("=" * 50)
    print(f"Images: {stats['total_images']}")
    print(f"Labels: {stats['total_labels']}")
    print(f"Boxes:  {stats['total_boxes']}")
    print(f"Empty labels: {stats['empty_labels']}")
    print(f"Issues: {len(issues)}")
    print("-" * 50)

    if issues:
        for issue in issues[:20]:
            print(issue)
        if len(issues) > 20:
            print(f"... and {len(issues) - 20} more issues")
    else:
        print("All checks passed!")

    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/processed", help="Dataset root directory")
    args = parser.parse_args()
    validate_dataset(args.dir)
