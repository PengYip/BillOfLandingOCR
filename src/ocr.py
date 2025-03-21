import os
import cv2
import numpy as np
from PIL import Image
from image_preprocessor import ImagePreprocessor

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化OCR和图像预处理器
ocr = PaddleOCR(use_angle_cls=False, lang="ch")
preprocessor = ImagePreprocessor()

def process_image(img_path, remove_watermark=False, watermark_params=None, enhance_params=None):
    """处理图像，支持去水印和图像增强
    Args:
        img_path: 图像路径
        remove_watermark: 是否去除水印
        watermark_params: 去水印参数
        enhance_params: 图像增强参数
    Returns:
        OCR结果和处理后的图像
    """
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 图像预处理
    if remove_watermark or enhance_params:
        image = preprocessor.preprocess_image(
            image,
            remove_watermark_params=watermark_params if remove_watermark else None,
            enhance_params=enhance_params
        )

    # 执行OCR
    result = ocr.ocr(image, cls=False)
    
    return result, image

# 示例：处理带水印的图像
img_path = os.path.join(current_dir, "test_img", "1.jpg")

# 设置去水印和增强参数
watermark_params = {
    'brightness_threshold': 200,
    'saturation_threshold': 30,
    'contrast_alpha': 1.3,
    'contrast_beta': 0
}

enhance_params = {
    'brightness': 1.0,
    'contrast': 1.2,
    'sharpness': 1.5
}

# 处理图像
result, processed_image = process_image(
    img_path,
    remove_watermark=True,
    watermark_params=watermark_params,
    enhance_params=enhance_params
)
print("Result structure:")
for idx, line in enumerate(result):
    print(f"Line {idx}:", line)
    print(f"Type of line[1]: {type(line[1])}")
    print(f"Content of line[1]: {line[1]}")
# 可视化结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]
im_show = draw_ocr(image, boxes, txts, scores, font_path=os.path.join(current_dir, 'simfang.ttf'))
im_show = Image.fromarray(im_show)
im_show.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读入图像,三通道
image=cv2.imread(os.path.join(current_dir, "test_img", "3.jpg"),cv2.IMREAD_COLOR) #timg.jpeg

#获得三个通道
Bch,Gch,Rch=cv2.split(image)

#保存三通道图片
cv2.imwrite(os.path.join(current_dir, 'blue_channel.jpg'),Bch)
cv2.imwrite(os.path.join(current_dir, 'green_channel.jpg'),Gch)
cv2.imwrite(os.path.join(current_dir, 'red_channel.jpg'),Rch)

import numpy as np
import cv2


img_path = os.path.join(current_dir, 'red_channel.jpg')
result = ocr.ocr(img_path, cls=False)

# 可视化结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]
im_show = draw_ocr(image, boxes, txts, scores, font_path=os.path.join(current_dir, 'simfang.ttf'))
im_show = Image.fromarray(im_show)
vis = np.array(im_show)
im_show.show()