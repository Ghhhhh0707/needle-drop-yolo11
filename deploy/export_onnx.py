"""
模型导出脚本：PyTorch -> ONNX / OpenVINO
"""
from pathlib import Path
from ultralytics import YOLO


def export_model(weights_path, format="onnx", imgsz=640):
    """
    导出模型到指定格式
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    print(f"加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"开始导出: format={format}, imgsz={imgsz}")
    model.export(format=format, imgsz=imgsz, opset=12)

    # 输出路径提示
    out_suffix = ".onnx" if format == "onnx" else "_openvino_model"
    print(f"导出完成: {weights_path.stem}{out_suffix}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX/OpenVINO")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "openvino"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    export_model(args.weights, args.format, args.imgsz)
