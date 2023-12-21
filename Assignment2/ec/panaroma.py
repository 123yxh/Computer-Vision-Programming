import numpy as np
import cv2
#Import necessary functions
import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../python'))
sys.path.append(module_path)
from planarH import computeH_ransac, compositeH
import matplotlib.pyplot as plt
from matchPics import matchPics

#读取图片
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

#获取形状
imH1, imW1, _ = left_img.shape
imH2, imW2, _ = right_img.shape
width = round(max(imW1, imW2)*1.2)

#图像边界填充-黑色
im2 = cv2.copyMakeBorder(right_img, 0, imH2 - imH1, width-imW2,
                         0, cv2.BORDER_CONSTANT, 0)

#特征匹配
matches, locs1, locs2 = matchPics(left_img, im2)
locs1 = locs1[matches[:, 0], 0:2]
locs2 = locs2[matches[:, 1], 0:2]

#计算单应矩阵的RANSAC
bestH2to1, inliers = computeH_ransac(locs1, locs2)

#建一个复合图像，将模板图像叠加在输入图像之上
pano_im = compositeH(bestH2to1, left_img, im2)

#像素比较，选取最大值
pano_im = np.maximum(im2, pano_im)

cv2.imwrite('pin_jie_result.png', pano_im)


#Write script for Q4.2x
