"""
数据集图片批量重命名脚本
功能：将原始图片统一重命名为 001.jpg / 002.png 等编号格式
"""
import shutil
import argparse
from pathlib import Path


def rename_images(image_dir, prefix="", start=1):
    """
    按顺序编号重命名图片
    """
    image_dir = Path(image_dir)
    files = sorted([
        f for f in image_dir.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])

    renamed = []
    for idx, old_path in enumerate(files, start):
        new_name = f"{prefix}{idx:03d}{old_path.suffix.lower()}"
        new_path = old_path.parent / new_name

        if old_path.name != new_name:
            shutil.move(str(old_path), str(new_path))
            renamed.append((old_path.name, new_name))
            print(f"{old_path.name} -> {new_name}")

    print(f"\n共处理 {len(files)} 张，重命名 {len(renamed)} 张")
    return renamed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rename images to sequential numbers")
    parser.add_argument("--dir", type=str, default="data/raw", help="Image directory")
    parser.add_argument("--prefix", type=str, default="", help="Filename prefix")
    parser.add_argument("--start", type=int, default=1, help="Start number")
    args = parser.parse_args()

    rename_images(args.dir, args.prefix, args.start)
