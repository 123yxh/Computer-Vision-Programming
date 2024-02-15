import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from getHarrisPoints import get_harris_points
from getRandomPoints import  get_random_points


# Load the uploaded image
image_path = '../data/campus/sun_acfgwrhgcjrpcbru.jpg'
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Define parameters
alpha = 1000  # number of points
k = 0.04     # Harris corner parameter

# Get random points
random_points = get_random_points(image, alpha)
print(random_points)
y_coords, x_coords = random_points[0][0], random_points[0][1]
# Get Harris points
harris_points = get_harris_points(image, alpha, k)
print(harris_points)
# 将列表转换为NumPy数组
harris_points_np = np.array(harris_points)
# Plot the original image and the points
plt.figure(figsize=(10, 8))
plt.imshow(image)

# Plot random points in blue
plt.scatter(x_coords, y_coords, c='blue', s=10, label='Random Points')
# Plot Harris points in red
plt.scatter(harris_points_np[:, 1], harris_points_np[:, 0], c='red', s=10, label='Harris Points')
# Show the plot with legends
plt.legend()
plt.axis('off')
plt.show()