import numpy as np
import cv2


def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	# 获取点数
	N = x1.shape[0]

	# 获取对应的矩阵向量
	u = x1[:, 0].reshape(N, 1)
	v = x1[:, 1].reshape(N, 1)

	x = x2[:, 0].reshape(N, 1)
	y = x2[:, 1].reshape(N, 1)

	# 构建系数矩阵A
	top = np.concatenate((x, y, np.ones((N, 1)), np.zeros((N, 3)), -np.multiply(x, u), -np.multiply(y, u), -u), axis=1)
	down = np.concatenate((np.zeros((N, 3)), x, y, np.ones((N, 1)), -np.multiply(x, v), -np.multiply(y, v), -1 * v),
						  axis=1)
	A = np.concatenate((top, down), axis=0)

	# 奇异值分解
	u, s, vh = np.linalg.svd(A)
	h = vh[-1, :]

	# 重塑为3x3的矩阵
	H2to1 = h.reshape(3, 3) / vh[-1, -1]

	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	# Compute the centroid of the points
	mean_x1 = np.mean(x1[:, 0], axis=0)
	mean_y1 = np.mean(x1[:, 1], axis=0)
	mean_x2 = np.mean(x2[:, 0], axis=0)
	mean_y2 = np.mean(x2[:, 1], axis=0)

	# Shift the origin of the points to the centroid
	x1_shifted = x1 - np.array([mean_x1, mean_y1])
	x2_shifted = x2 - np.array([mean_x2, mean_y2])

	# Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1_max = np.amax(abs(x1_shifted), axis=0)
	x2_max = np.amax(abs(x2_shifted), axis=0)

	x1_norm = np.divide(x1_shifted, x1_max)
	x2_norm = np.divide(x2_shifted, x2_max)

	# Similarity transform 1
	T1 = np.array(([1 / x1_max[0], 0, -mean_x1[0] / x1_max[0]], [0, 1 / x1_max[1], -mean_x1[1] / x1_max[1]], [0, 0, 1]))

	# Similarity transform 2
	T2 = np.array(([1 / x2_max[0], 0, -mean_x2[0] / x2_max[0]], [0, 1 / x2_max[1], -mean_x2[1] / x2_max[1]], [0, 0, 1]))

	# Compute homography
	H_prime = computeH(x1_norm, x2_norm)

	# Denormalization
	H2to1 = np.linalg.inv(T2) @ H_prime @ T1
	

	return H2to1




def computeH_ransac(x1, x2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	N = x1.shape[0]

	# set RANSAC parameter
	max_iters = 50  # 最大迭代次数
	inlier_tol = 2  # 内点阈值，单位是像素
	max_inlier = -1  # 最大内点数，初始值为-1

	match_x1 = np.hstack((x1, np.ones((N, 1))))  # u,v,1
	match_x2 = np.hstack((x2, np.ones((N, 1))))  # x,y,1

	# 对每次迭代进行循环
	for i in range(max_iters):

		# 随机选取四个点作为样本
		indices = np.random.randint(N, size=4)
		sample_1 = x1[indices]
		sample_2 = x2[indices]

		# 使用 computeH_norm 函数计算样本点对之间的单应矩阵 H
		H = computeH(sample_1, sample_2)

		match = np.matmul(H, match_x2.T)
		match = match.T

		div = np.expand_dims(match[:, -1], axis=1)
		diff = ((match / div) - match_x1)

		# 归一化
		diff = np.linalg.norm(diff, axis=1)
		inlier_calcim = np.where(diff < inlier_tol, 1, 0)

		# 如果内点的个数大于当前的最大内点数，就更新最佳单应矩阵和内点向量
		if (np.sum(inlier_calcim) > max_inlier):
			max_inlier = np.sum(inlier_calcim)  # 统计内点个数
			inliers = inlier_calcim

	# 找到元素等于1的索引
	ind_use = np.where(inliers == 1)
	# 计算单应矩阵
	bestH2to1 = computeH(x1[ind_use[0], :], x2[ind_use[0], :])


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	# Create mask of same size as template
	mask = np.ones((template.shape[0], template.shape[1], 3), dtype=np.uint8)
	# Warp mask by appropriate homography
	warp_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))
	# cv2.imwrite('../data/Mask.jpg',warp_mask)

	# Warp template by appropriate homography
	warp_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
	# cv2.imwrite('../data/warp_template.jpg',warp_template)
	inverted_warp_mask = np.where(warp_mask >= 1, 0, 1).astype(dtype=np.uint8)

	# Use mask to combine the warped template and the image
	cv_desk_cut = np.multiply(img, inverted_warp_mask)
	composite_img = cv_desk_cut + warp_template
	
	return composite_img


