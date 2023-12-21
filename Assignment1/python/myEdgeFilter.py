import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil, atan2, degrees
from myImageFilter import myImageFilter
import cv2

def myEdgeFilter(input_image, sigma_value):
    # Create Gaussian filter
    filter_size = 2 * ceil(3 * sigma_value) + 1
    gaussian_filter = signal.gaussian(filter_size, sigma_value).reshape(filter_size, 1)
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

    smoothed_image = myImageFilter(input_image, gaussian_filter)
    smoothed_image = myImageFilter(smoothed_image, gaussian_filter.T)

    # Sobel filters
    sobel_x_filter = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_y_filter = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    # Compute gradients in x and y directions of the image
    gradient_x = myImageFilter(smoothed_image, sobel_x_filter)
    gradient_y = myImageFilter(smoothed_image, sobel_y_filter)

    # Compute magnitude and angle of gradients
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    print(np.shape(angle))

    print(angle)
    M, N = input_image.shape
    output_image = np.zeros((M, N))
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]

                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    output_image[i, j] = magnitude[i, j]
                else:
                    output_image[i, j] = 0
            except IndexError as e:
                pass

    return output_image

# file_path = '../data/img01.jpg'  # 替换为你的图像文件路径
# img0 = load_image(file_path)
#
# # 调用函数
# sigma = 2
# img1= myEdgeFilter(img0, sigma)
#
# # 显示结果
# plt.imshow(img1, cmap='gray')
# plt.title('Edge Magnitude')
# plt.show()
