import cv2
import numpy as np
from typing import Tuple, Optional

class ImagePreprocessor:
    def __init__(self):
        pass

    def remove_watermark(self, image: np.ndarray,
                        brightness_threshold: int = 200,
                        saturation_threshold: int = 30,
                        contrast_alpha: float = 1.3,
                        contrast_beta: int = 0) -> np.ndarray:
        """
        去除图像中的水印
        Args:
            image: 输入图像（BGR格式）
            brightness_threshold: 亮度阈值，用于识别水印区域
            saturation_threshold: 饱和度阈值，用于识别水印区域
            contrast_alpha: 对比度增强系数
            contrast_beta: 亮度调整值
        Returns:
            处理后的图像
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 基于亮度和饱和度识别水印区域
        mask = np.logical_and(v > brightness_threshold, s < saturation_threshold)
        mask = mask.astype(np.uint8) * 255

        # 对掩码进行形态学操作，连接相近的区域
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 创建反掩码
        inv_mask = cv2.bitwise_not(mask)

        # 应用掩码去除水印
        result = cv2.bitwise_and(image, image, mask=inv_mask)

        # 增强对比度
        result = cv2.convertScaleAbs(result, alpha=contrast_alpha, beta=contrast_beta)

        return result

    def enhance_image(self, image: np.ndarray,
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     sharpness: float = 1.0) -> np.ndarray:
        """
        增强图像质量
        Args:
            image: 输入图像
            brightness: 亮度调整系数
            contrast: 对比度调整系数
            sharpness: 锐化程度
        Returns:
            增强后的图像
        """
        # 亮度和对比度调整
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        # 锐化处理
        if sharpness > 1.0:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * sharpness
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def preprocess_image(self, image: np.ndarray,
                        remove_watermark_params: Optional[dict] = None,
                        enhance_params: Optional[dict] = None) -> np.ndarray:
        """
        图像预处理主函数
        Args:
            image: 输入图像
            remove_watermark_params: 去水印参数字典
            enhance_params: 图像增强参数字典
        Returns:
            处理后的图像
        """
        result = image.copy()

        # 去除水印
        if remove_watermark_params is not None:
            result = self.remove_watermark(result, **remove_watermark_params)

        # 图像增强
        if enhance_params is not None:
            result = self.enhance_image(result, **enhance_params)

        return result