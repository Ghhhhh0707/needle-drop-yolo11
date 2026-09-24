"""
红圈标注 → YOLO TXT 导出
功能：识别图片中的红色圆圈/标记，转换为YOLO格式边界框
"""
import os
import cv2
import numpy as np
from pathlib import Path


def extract_red_circles(image_path, visualize=False):
    """
    从图片中提取红色标记区域，返回检测框列表 [(x, y, w, h), ...]
    坐标为绝对像素值
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色在HSV中分为两段：低 hue 和高 hue
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 过滤太小的噪声
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return img, boxes


def convert_to_yolo(img_width, img_height, x, y, w, h):
    """
    将绝对坐标转换为YOLO归一化格式
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height

    # 裁剪到 [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return x_center, y_center, nw, nh


def export_annotations(image_dir, output_label_dir, class_id=0, visualize_dir=None):
    """
    批量导出标注
    """
    image_dir = Path(image_dir)
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if visualize_dir:
        visualize_dir = Path(visualize_dir)
        visualize_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(ext))

    for img_path in image_paths:
        try:
            img, boxes = extract_red_circles(img_path)
        except Exception as e:
            print(f"[错误] {img_path}: {e}")
            continue

        h, w = img.shape[:2]
        label_lines = []

        for (bx, by, bw, bh) in boxes:
            xc, yc, nw, nh = convert_to_yolo(w, h, bx, by, bw, bh)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # 写入TXT
        txt_name = img_path.stem + ".txt"
        txt_path = output_label_dir / txt_name
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        # 可视化校验图
        if visualize_dir:
            vis = img.copy()
            for (bx, by, bw, bh) in boxes:
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            vis_path = visualize_dir / img_path.name
            cv2.imwrite(str(vis_path), vis)

        print(f"[导出] {img_path.name}: {len(boxes)} 个框 -> {txt_path}")

    print("\n导出完成。请人工抽检 visualize/ 目录下的预览图！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export red-circle annotations to YOLO TXT")
    parser.add_argument("--images", type=str, default="data/raw", help="Input image directory")
    parser.add_argument("--labels", type=str, default="data/labels/all", help="Output label directory")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID")
    parser.add_argument("--viz", type=str, default="visualize", help="Visualization output dir")
    args = parser.parse_args()

    export_annotations(args.images, args.labels, args.class_id, args.viz)
