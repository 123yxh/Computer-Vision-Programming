import cv2
import numpy as np
import math
from myEdgeFilter import myEdgeFilter
from PIL import Image

# rhoRes    = 1.5
# thetaRes  = np.pi / 180

import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    # Get the number of rows and columns in the image
    rows, cols = img_threshold.shape

    # Calculate the maximum possible ρ value, which is the length of the image diagonal
    max_rho = np.ceil(np.hypot(rows, cols))

    # Generate arrays for ρ and θ values to return
    rhos = np.arange(0, max_rho, rhoRes)
    thetas = np.arange(0, 2 * np.pi, thetaRes)

    # Create a Hough transform accumulator, initialized with zeros
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int64)

    # Get the coordinates of edge points in the image
    y_coords, x_coords = np.nonzero(img_threshold)

    # Vote for each edge point
    for x, y in zip(x_coords, y_coords): # Iterate over each edge point
        for k in range(len(thetas)): # Iterate over each θ value
            theta = thetas[k] # Get θ value
            rho = x * np.cos(theta) + y * np.sin(theta) # Calculate ρ value
            if rho > 0: # If ρ value is greater than 0
                n = int(np.round(rho / rhoRes)) # Calculate the index corresponding to ρ value
                accumulator[n][k] += 1 # Increment the accumulator at the corresponding position

    # Return the result
    return accumulator, rhos, thetas




# file_path = './Assignment 1/data/img01.jpg'
#
#
# def load_image(file_path):
#     image = Image.open(file_path)
#     image = image.convert('L')  # 将图像转换为灰度模式
#     image_array = np.array(image)/255   #归一化[0.255]
#     return image_array
#
#
# img0 = load_image(file_path)
# img_edge = myEdgeFilter(img0, 2)
# img_threshold = np.float32(img_edge > 0.1)
# # Using scipy.signal.gaussian to get the Gaussian filter
# [img_hough, rhoScale, thetaScale] = myHoughTransform(img_threshold, \
#                                                      rhoRes, thetaRes)
# # Normalize the hough image for display
# img_hough_normalized = cv2.normalize(img_hough, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imshow("hough Image", img_hough_normalized)
# cv2.imshow("edge Image", img_edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()