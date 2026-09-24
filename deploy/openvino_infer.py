"""
OpenVINO 推理脚本
支持：单图、文件夹批量、摄像头
"""
import time
import cv2
import numpy as np
from pathlib import Path


def preprocess(image, input_size=(640, 640)):
    """
    预处理：resize, normalize, transpose, expand_dims
    """
    h, w = image.shape[:2]
    # 等比缩放+灰边填充
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 灰边填充
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # normalize to [0,1] and NCHW
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    return blob, scale, (top, left)


def postprocess(outputs, conf_thresh=0.25, iou_thresh=0.45, scale=1.0, pad=(0, 0)):
    """
    后处理：解析输出，NMS，坐标还原
    注意：此处为通用骨架，具体实现需根据导出模型的输出格式调整
    """
    # TODO: 根据实际ONNX/OpenVINO输出格式完成解析
    # Ultralytics YOLOv8/v11 导出格式通常为: [batch, 84, num_anchors] 或 [batch, num_anchors, 84]
    # 84 = 4(box) + 80(coco classes); 单类时应为 5
    return []


def draw_boxes(image, boxes, color=(0, 255, 0)):
    """
    在图上画框
    """
    for (x1, y1, x2, y2, conf, cls) in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"defect {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def infer_image(model_path, image_path, conf_thresh=0.25, save_path=None):
    """
    单图推理
    """
    try:
        from openvino.runtime import Core
    except ImportError:
        print("[错误] 未安装 openvino，请先: pip install openvino")
        return

    ie = Core()
    model = ie.read_model(model_path)
    compiled = ie.compile_model(model, "CPU")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return

    blob, scale, pad = preprocess(image)

    t0 = time.time()
    outputs = compiled([blob])
    infer_time = time.time() - t0

    # TODO: 完成后处理并输出结果
    print(f"推理耗时: {infer_time*1000:.1f}ms, FPS: {1/infer_time:.1f}")

    if save_path:
        cv2.imwrite(str(save_path), image)
        print(f"结果保存: {save_path}")


def infer_folder(model_path, image_dir, output_dir, conf_thresh=0.25):
    """
    文件夹批量推理
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_dir.glob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        save_path = output_dir / img_path.name
        infer_image(model_path, img_path, conf_thresh, save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenVINO inference for YOLO11")
    parser.add_argument("--model", type=str, required=True, help="Path to .xml or .onnx")
    parser.add_argument("--source", type=str, required=True, help="Image path or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    src = Path(args.source)
    if src.is_dir():
        infer_folder(args.model, src, args.save, args.conf)
    else:
        infer_image(args.model, src, args.conf, args.save)
