import numpy as np
import cv2

def myHoughLines(img_hough, nLines):
    # 归一化累计矩阵
    img_hough = cv2.normalize(img_hough, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # 1. 非极大值抑制
    # 使用3x3的结构元素进行膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img_hough, kernel)

    # 膨胀后的值应该是邻域中的最大值，只有当原图中的值与膨胀后的值相同且大于0时，它才是局部最大值
    local_maxima = (img_hough == dilated) * (img_hough > 0)

    # 2. 选择nLines个最大的峰值
    # 获取峰值的坐标
    y_coords, x_coords = np.where(local_maxima)
    peak_values = img_hough[local_maxima]

    # 获取最大的nLines个峰值的索引
    sorted_idxs = np.argsort(peak_values)[::-1][:nLines]
    rhos = y_coords[sorted_idxs]
    thetas = x_coords[sorted_idxs]

    return rhos, thetas