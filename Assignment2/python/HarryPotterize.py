import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
from planarH import compositeH
import matplotlib.pyplot as plt


#读取图片
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover=cv2.imread('../data/hp_cover.jpg')

#匹配特征
matches1,locs1,locs2=matchPics(cv_desk,cv_cover)

#交换数组中特定列的值
locs1[:,[0,1]] = locs1[:,[1,0]]
locs2[:,[0,1]] = locs2[:,[1,0]]

#计算单应矩阵的RANSAC
bestH2to1,inliers=computeH_ransac(locs1[matches1[:,0]],locs2[matches1[:,1]])

#将图像填充到相同的空间
dim_1=(cv_cover.shape[1],cv_cover.shape[0])
hp_cover=cv2.resize(hp_cover,dim_1)

#建一个复合图像，将模板图像叠加在输入图像之上
composite_img=compositeH(bestH2to1,hp_cover ,cv_desk)
print("Shape_of_composite_image:",composite_img.shape)

cv2.imwrite('../data/composite_img.jpg',composite_img)


#Write script for Q3.9
