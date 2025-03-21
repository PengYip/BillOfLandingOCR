import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_angle_cls=False, lang="ch") 
img_path = os.path.join(current_dir, "test_img", "1.jpg")
result = ocr.ocr(img_path, cls=False)
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