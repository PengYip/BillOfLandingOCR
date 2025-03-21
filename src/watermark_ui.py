import gradio as gr
import cv2
import numpy as np
from image_preprocessor import ImagePreprocessor
import os

# 初始化图像预处理器
preprocessor = ImagePreprocessor()

def process_image(image,
                brightness_threshold,
                saturation_threshold,
                contrast_alpha,
                contrast_beta,
                brightness,
                contrast,
                sharpness):
    """处理图像并返回原图和处理后的图像"""
    # 设置去水印参数
    watermark_params = {
        'brightness_threshold': int(brightness_threshold),
        'saturation_threshold': int(saturation_threshold),
        'contrast_alpha': float(contrast_alpha),
        'contrast_beta': int(contrast_beta)
    }
    
    # 设置图像增强参数
    enhance_params = {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'sharpness': float(sharpness)
    }
    
    # 处理图像
    processed = preprocessor.preprocess_image(
        image,
        remove_watermark_params=watermark_params,
        enhance_params=enhance_params
    )
    
    # 返回原图和处理后的图像
    return [image, processed]

# 创建Gradio界面
with gr.Blocks(title="水印去除与图像增强工具") as demo:
    gr.Markdown("## 水印去除与图像增强工具")
    
    with gr.Row():
        # 输入图像
        input_image = gr.Image(label="上传图片", type="numpy")
        
        # 显示原图和处理后的图像
        output_images = gr.Gallery(label="处理结果", show_label=True, elem_id="gallery", columns=2, height="auto")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 去水印参数")
            brightness_threshold = gr.Slider(minimum=0, maximum=255, value=200, step=1, label="亮度阈值")
            saturation_threshold = gr.Slider(minimum=0, maximum=255, value=30, step=1, label="饱和度阈值")
            contrast_alpha = gr.Slider(minimum=0.1, maximum=3.0, value=1.3, step=0.1, label="对比度增强系数")
            contrast_beta = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="亮度调整值")
        
        with gr.Column():
            gr.Markdown("### 图像增强参数")
            brightness = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="亮度")
            contrast = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="对比度")
            sharpness = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="锐化程度")
    
    # 处理按钮
    process_btn = gr.Button("处理图像")
    process_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            brightness_threshold,
            saturation_threshold,
            contrast_alpha,
            contrast_beta,
            brightness,
            contrast,
            sharpness
        ],
        outputs=output_images
    )

if __name__ == "__main__":
    demo.launch()