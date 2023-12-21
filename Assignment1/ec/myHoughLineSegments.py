import numpy as np
import math
import cv2


def hough_lines_p(image, theta_res, rho_res):
    # 图像的高度和宽度
    height, width = image.shape
    # 最大极径
    max_rho = int(math.hypot(height, width))
    # 极角范围
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    # 极径和极角的范围
    rhos = np.arange(-max_rho, max_rho, rho_res)
    # 构建累加器
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # 遍历图像像素，检测边缘点
    edge_points = np.argwhere(image > 0)

    for y, x in edge_points:
        for theta_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1

    # 根据累加器中的值找到直线
    lines = []
    for i in range(len(rhos)):
        for j in range(len(thetas)):
            if accumulator[i, j] > 142:
                rho = rhos[i]
                theta = thetas[j]
                lines.append((rho, theta))

    return lines


# 请将图像加载到"image"变量中，确保它是灰度图像

# 设置参数
theta_res = 1  # 极角分辨率
rho_res = 1  # 极径分辨
threshold = 165  # 累加器阈值

input_image = cv2.imread('../data/img03.jpg')
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# 进行边缘检测
edges = cv2.Canny(gray, 50, 160, apertureSize=3)

# 运行Hough变换
detected_lines = hough_lines_p(edges, theta_res, rho_res)

# 输出检测到的线条
for rho, theta in detected_lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    cv2.line(input_image , (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示带有检测线条的图像
cv2.imshow("edges_Imge", edges )
cv2.imshow("Hough Lines", input_image )
cv2.waitKey(0)
cv2.destroyAllWindows()
