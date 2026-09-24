"""
Gradio Demo 界面
支持：上传图片检测、摄像头实时检测
"""
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 加载模型（启动时加载，避免每次推理重复初始化）
MODEL_PATH = "runs/baseline_n/exp1/weights/best.pt"  # 可改为ONNX/OpenVINO路径
model = None


def load_model(path=MODEL_PATH):
    global model
    if model is None:
        print(f"加载模型: {path}")
        model = YOLO(path)
    return model


def detect_image(image):
    """
    图片检测：输入 PIL.Image 或 numpy 数组，返回标注后的图 + 文字结果
    """
    if image is None:
        return None, "未收到图片"

    m = load_model()
    results = m.predict(source=image, conf=0.25, verbose=False)

    # 绘制结果
    annotated = results[0].plot()
    annotated_rgb = annotated[..., ::-1]  # BGR to RGB

    # 统计信息
    boxes = results[0].boxes
    count = len(boxes)
    confs = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
    info = f"检测到缺陷: {count} 个\n"
    if confs:
        info += "置信度: " + ", ".join([f"{c:.2f}" for c in confs])
    else:
        info += "无缺陷"

    return Image.fromarray(annotated_rgb), info


def build_ui():
    with gr.Blocks(title="漏针检测 Demo") as demo:
        gr.Markdown("# 电脑横机织布漏针智能检测 (YOLO11)")
        gr.Markdown("上传布面照片，系统自动识别漏针/破洞缺陷。")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="上传图片")
                btn = gr.Button("开始检测", variant="primary")
            with gr.Column():
                output_img = gr.Image(type="pil", label="检测结果")
                output_txt = gr.Textbox(label="检测信息", lines=3)

        btn.click(fn=detect_image, inputs=input_img, outputs=[output_img, output_txt])

        gr.Markdown("---")
        gr.Markdown("💡 提示：模型为单类别 `defect`（漏针/破洞），置信度阈值 0.25")

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    if args.model:
        MODEL_PATH = args.model
        load_model(MODEL_PATH)

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
