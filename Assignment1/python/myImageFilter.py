import numpy as np
from PIL import Image
import cv2
from scipy.signal import gaussian, convolve2d

# def load_image(file_path):
#     image = Image.open(file_path)
#     image = image.convert('L')  # 将图像转换为灰度模式
#     image_array = np.array(image)/255   #归一化[0.255]
#     return image_array

def myImageFilter(img0, h):
    # 获取图像和滤波器的尺寸
    img_height, img_width = img0.shape
    filter_height, filter_width = h.shape

    # 计算零填充的大小
    pad_height = filter_height // 2
    pad_width = filter_width // 2

    # 创建一个全零的输出图像
    img1 = np.zeros_like(img0)

    # 零填充图像
    padded_img = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    # 进行卷积
    for i in range(img_height):
        for j in range(img_width):
            # 提取与滤波器大小相匹配的图像块
            image_patch = padded_img[i:i+filter_height, j:j+filter_width]

            # 计算卷积
            img1[i, j] = np.sum(image_patch * h)

    return img1

# 读取图像文件
# file_path = '../data/img01.jpg'  # 替换为你的图像文件路径
# img0 = load_image(file_path)
#
# h = np.array([[-1, 0, 1],
#               [-1, 0, 1],
#               [-1, 0, 1]])
#
# result = myImageFilter(img0, h)
# print(result)
# #调用卷积算子
# img1 = cv2.filter2D(img0, -1,h)
#
# # 显示原始图像和滤波后的图像
# cv2.imshow('Original Image', img0)
# cv2.imshow('Filtered Image', result)
# cv2.imshow('convolve2d Image', img1)
#
# # 等待用户按下任意键后关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

