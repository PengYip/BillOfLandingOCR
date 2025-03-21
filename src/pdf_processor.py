import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Optional, Tuple
from image_preprocessor import ImagePreprocessor

class PDFProcessor:
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """
        将PDF文件转换为图像列表
        Args:
            pdf_path: PDF文件路径
            dpi: 转换分辨率
        Returns:
            图像列表（BGR格式）
        """
        # 转换PDF为PIL Image列表
        pil_images = convert_from_path(pdf_path, dpi=dpi)
        
        # 转换为OpenCV格式（BGR）
        cv_images = []
        for pil_image in pil_images:
            # 转换为RGB格式的numpy数组
            rgb_image = np.array(pil_image)
            # 转换为BGR格式
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv_images.append(bgr_image)
        
        return cv_images

    def images_to_pdf(self, images: List[np.ndarray], output_path: str):
        """
        将图像列表保存为PDF文件
        Args:
            images: 图像列表（BGR格式）
            output_path: 输出PDF文件路径
        """
        pil_images = []
        for image in images:
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_image = Image.fromarray(rgb_image)
            pil_images.append(pil_image)
        
        # 保存为PDF
        if pil_images:
            pil_images[0].save(
                output_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=pil_images[1:]
            )

    def process_pdf(self, 
                    input_path: str,
                    output_path: str,
                    remove_watermark_params: Optional[dict] = None,
                    enhance_params: Optional[dict] = None,
                    dpi: int = 200) -> str:
        """
        处理PDF文件，支持去水印和图像增强
        Args:
            input_path: 输入PDF文件路径
            output_path: 输出PDF文件路径
            remove_watermark_params: 去水印参数
            enhance_params: 图像增强参数
            dpi: PDF转图像的分辨率
        Returns:
            处理后的PDF文件路径
        """
        # 将PDF转换为图像
        images = self.pdf_to_images(input_path, dpi)
        
        # 处理每一页图像
        processed_images = []
        for image in images:
            processed = self.image_preprocessor.preprocess_image(
                image,
                remove_watermark_params=remove_watermark_params,
                enhance_params=enhance_params
            )
            processed_images.append(processed)
        
        # 保存为PDF
        self.images_to_pdf(processed_images, output_path)
        
        return output_path