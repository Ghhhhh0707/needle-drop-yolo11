"""
数据集去重脚本
功能：MD5 + pHash 去重，输出重复对清单
"""
import os
import hashlib
import json
from pathlib import Path
from PIL import Image
import imagehash


def md5_file(path):
    """计算文件MD5"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(image_dir, out_json="dedup_results.json"):
    """
    扫描目录中的图片，按MD5和感知哈希(pHash)去重
    """
    image_dir = Path(image_dir)
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(image_dir.rglob(ext))

    md5_map = {}
    phash_map = {}
    duplicates = []

    for f in files:
        try:
            md5 = md5_file(f)
            phash = str(imagehash.phash(Image.open(f)))
        except Exception as e:
            print(f"[跳过] {f}: {e}")
            continue

        key = (md5, phash)
        if key in md5_map:
            dup = {
                "file1": str(md5_map[key]),
                "file2": str(f),
                "md5": md5,
                "phash": phash,
            }
            duplicates.append(dup)
            print(f"[重复] {md5_map[key]} <=> {f}")
        else:
            md5_map[key] = f
            phash_map[str(f)] = phash

    result = {
        "total_scanned": len(files),
        "unique_files": len(md5_map),
        "duplicate_pairs": duplicates,
    }

    out_path = Path(out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n扫描完成：共 {len(files)} 张，唯一 {len(md5_map)} 张，重复对 {len(duplicates)} 组")
    print(f"结果保存至：{out_path.absolute()}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset deduplication via MD5 + pHash")
    parser.add_argument("--dir", type=str, default="data/raw", help="Image directory to scan")
    parser.add_argument("--out", type=str, default="dedup_results.json", help="Output JSON path")
    args = parser.parse_args()

    find_duplicates(args.dir, args.out)
