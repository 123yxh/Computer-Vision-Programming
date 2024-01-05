import numpy as np
import helper as hp
from scipy.linalg import svd
from scipy.linalg import inv 
from scipy import linalg
from scipy import signal  
# from hp import refineF
"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""

def eight_point(pts1, pts2, M):
    # Scale the input points
    pts1_scaled, pts2_scaled = pts1 / M, pts2 / M

    # Prepare matrix A for the eight-point algorithm
    num_points = pts1_scaled.shape[0]
    A = np.zeros((num_points, 9))

    # Efficiently populate matrix A using numpy broadcasting
    A[:, 0] = pts2_scaled[:, 0] * pts1_scaled[:, 0]  # x2' * x1'
    A[:, 1] = pts2_scaled[:, 0] * pts1_scaled[:, 1]  # x2' * y1'
    A[:, 2] = pts2_scaled[:, 0]  # x2'
    A[:, 3] = pts2_scaled[:, 1] * pts1_scaled[:, 0]  # y2' * x1'
    A[:, 4] = pts2_scaled[:, 1] * pts1_scaled[:, 1]  # y2' * y1'
    A[:, 5] = pts2_scaled[:, 1]  # y2'
    A[:, 6] = pts1_scaled[:, 0]  # x1'
    A[:, 7] = pts1_scaled[:, 1]  # y1'
    A[:, 8] = 1

    # Perform Singular Value Decomposition (SVD) on A
    _, _, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)

    # Refine the fundamental matrix
    F_refined = hp.refineF(F, pts1_scaled, pts2_scaled)

    # Construct the scaling transformation matrix and unscale F
    T = np.diag([1 / M, 1 / M, 1])
    F_unscaled = T.T @ F_refined @ T

    return F_unscaled

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    step = 10       # 设置搜索步长
    sigma = 5       # 设置高斯滤波器的标准差
    correspondences = []

    for pt in pts1:
        # Compute the epipolar line
        P1 = np.array([pt[0], pt[1], 1])
        epline = F @ P1
        epline /= np.linalg.norm(epline)
        a, b, c = epline

        # Initialize minimum distance
        min_distance = np.inf
        x2, y2 = 0, 0

        # Extract patch from the first image
        x1, y1 = int(round(pt[0])), int(round(pt[1]))
        patch1 = im1[y1 - step:y1 + step + 1, x1 - step:x1 + step + 1]
        kernel = createGaussianFilter((2 * step + 1, 2 * step + 1), sigma)

        # Iterate over possible corresponding points in the second image
        for y in range(y1 - sigma * step, y1 + sigma * step):
            x = int(round((-b * y - c) / a))
            s_h, e_h = y - step, y + step + 1
            s_w, e_w = x - step, x + step + 1

            if 0 <= s_w < im2.shape[1] and 0 <= e_w < im2.shape[1] and 0 <= s_h < im2.shape[0] and 0 <= e_h < im2.shape[
                0]:
                patch2 = im2[s_h:e_h, s_w:e_w]

                # Calculate the weighted distance between patches
                weighted_distances = [np.linalg.norm(kernel * (patch1[:, :, i] - patch2[:, :, i])) for i in
                                      range(patch2.shape[2])]
                error = sum(weighted_distances)

                # Update the best match if a closer one is found
                if error < min_distance:
                    min_distance = error
                    x2, y2 = x, y

        correspondences.append([x2, y2])

    return np.array(correspondences)


def createGaussianFilter(shape, sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # Transpose the intrinsic matrix of the second camera
    K2_transposed = np.transpose(K2)
    # Left-multiply with the fundamental matrix F
    left_multiplied = np.dot(K2_transposed, F)
    # Right-multiply with the intrinsic matrix of the first camera
    essential_mat = np.dot(left_multiplied, K1)

    return essential_mat


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    num_points = pts1.shape[0]
    triangulated_points = np.zeros((num_points, 3))

    for i in range(num_points):
        # Construct matrix A for each point
        a1 = pts1[i, 0] * P1[2, :] - P1[0, :]
        a2 = pts1[i, 1] * P1[2, :] - P1[1, :]
        a3 = pts2[i, 0] * P2[2, :] - P2[0, :]
        a4 = pts2[i, 1] * P2[2, :] - P2[1, :]
        A = np.vstack((a1, a2, a3, a4))

        # Perform Singular Value Decomposition (SVD)
        _, _, v = np.linalg.svd(A)
        point_3D = v[-1]
        point_3D /= point_3D[3]
        triangulated_points[i, :] = point_3D[:3]

    return triangulated_points


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""

def camera_center(K, R, t):
    return -np.linalg.inv(K @ R) @ (K @ t)

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = camera_center(K1, R1, t1).reshape(3, 1)
    c2 = camera_center(K2, R2, t2).reshape(3, 1)

    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(r1.ravel(), R1[:, 2]).reshape(3, 1)
    r3 = np.cross(r2.ravel(), r1.ravel()).reshape(3, 1)

    R = np.hstack((r1, r2, r3))

    t1p = -R @ c1
    t2p = -R @ c2

    M1 = K2 @ R @ np.linalg.inv(K1 @ R1)
    M2 = K2 @ R @ np.linalg.inv(K2 @ R2)

    return M1, M2, K2, K2, R, R, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    img_height, img_width = im1.shape
    disparity_map = np.zeros((img_height, img_width))
    padded_img2 = np.pad(im2, ((0, 0), (max_disp, 0)), "constant")
    conv_window = np.ones((win_size, win_size))
    min_ssd = np.full((img_height, img_width), np.inf)

    for disp in range(max_disp + 1):
        shifted_img2 = padded_img2[:, max_disp - disp:img_width + max_disp - disp]
        squared_diff = np.square(shifted_img2 - im1)
        ssd_convolved = signal.convolve2d(squared_diff, conv_window, mode="same", boundary='symm')

        update_positions = ssd_convolved < min_ssd
        disparity_map[update_positions] = disp
        min_ssd[update_positions] = ssd_convolved[update_positions]

    return disparity_map

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = camera_center(K1, R1, t1)
    c2 = camera_center(K2, R2, t2)
    b = np.linalg.norm(c1 - c2)
    f = K1[0, 0]
    rows, cols = dispM.shape
    depth_map = np.zeros((rows, cols))

    for y in range(rows):
        for x in range(cols):
            disparity_value = dispM[y, x]
            if disparity_value != 0:
                depth = b * (f / disparity_value)
                depth_map[y, x] = depth

    return depth_map

"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # Number of points
    num_points = x.shape[0]

    # Prepare matrix A for the Direct Linear Transform
    matrixA = []
    for i in range(num_points):
        # World coordinates and image coordinates
        X_world = X[i, :3]  # Extract world coordinates (x, y, z)
        x_image = x[i, :2]  # Extract image coordinates (x, y)

        # Constructing the two rows per point as per DLT
        row1 = np.hstack([X_world, 1, np.zeros(4), -x_image[0] * X_world, -x_image[0]])
        row2 = np.hstack([np.zeros(4), X_world, 1, -x_image[1] * X_world, -x_image[1]])

        # Appending to matrix A
        matrixA.extend([row1, row2])

    # Singular Value Decomposition
    _, _, vh = np.linalg.svd(np.array(matrixA))

    # The solution is the last row of V (or last column of V transposed) reshaped to 3x4
    pose_matrix = vh[-1].reshape(3, 4)

    return pose_matrix


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # Perform Singular Value Decomposition (SVD) on matrix P
    _, _, vh = np.linalg.svd(P)
    c = vh[-1]
    newc = np.array([c[0]/c[3], c[1]/c[3], c[2]/c[3]])

    # Extract the first three columns of P as matrix M
    M = P[:, 0:3]

    # Perform QR decomposition on the flipped M to get RQ decomposition
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)
    Q = Q.T

    # Correct the signs to ensure the diagonal of R is positive
    T = np.diag(np.sign(np.diag(R)))
    R = R @ T
    K = T @ Q

    # Ensure the upper left 3x3 part of K is upper triangular
    K = np.flipud(K)

    # Calculate the translation vector t
    t = -M @ newc

    return K, R, t
