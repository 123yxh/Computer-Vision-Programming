import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 1. Load the two temple images and the points from data/some_corresp.npz
load_some_corresp = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
# 2. Run eight_point to compute F
N = load_some_corresp['pts1'].shape[0]
# print(data_some_corresp['pts1'])
h,w,d = im1.shape
M = max(load_some_corresp['pts1'].max(), load_some_corresp['pts2'].max())
print(load_some_corresp['pts1'].shape, load_some_corresp['pts2'].shape, M.shape)
F_eight_point = sub.eight_point(load_some_corresp['pts1'], load_some_corresp['pts2'], M)
try:
    hlp.displayEpipolarF(im1, im2, F_eight_point)
except Exception:
    pass
pts2=load_some_corresp['pts2']
# 3. Load points in image 1 from data/temple_coords.npz
load_temple_coords = np.load('../data/temple_coords.npz')
# 4. Run epipolar_correspondences to get points in image 2
a= sub.epipolar_correspondences(im1, im2, F_eight_point, load_temple_coords['pts1'])
try:
    hlp.epipolarMatchGUI(im1, im2, F_eight_point)
except Exception:
    pass

data_intrinsics = np.load('../data/intrinsics.npz')
E = sub.essential_matrix(F_eight_point,data_intrinsics['K1'], data_intrinsics['K2'])
print(E)
# 5. Compute the camera projection matrix P1
P1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
# print(P1)
# 6. Use camera2 to get 4 camera projection matrices P2
P2 = hlp.camera2(E)
print(P2)
best_matrix = None
max_points_in_front = 0
#Selected Matrix
for i in range(4):
    P2_test = P2[:,:,i]
    P2_temp = np.matmul(data_intrinsics['K2'], P2_test)
    points_3D = sub.triangulate(P1, load_temple_coords['pts1'], P2_test, a)

    num_points_in_front = 0
    for point in points_3D:
        # Make sure that the points are homogeneous coordinates
        point_homog = np.append(point, 1)  # 将点转换为齐次坐标 [x, y, z, 1]

        # Make sure that the matrix multiplication can be performed correctly
        if point_homog[2] > 0 and (P2_test @ point_homog)[2] > 0:
            num_points_in_front += 1
    #choose point
    if num_points_in_front > max_points_in_front:
        max_points_in_front = num_points_in_front
        best_matrix = P2_test

# 7. Run triangulate using the projection matrices
print(P1.shape)
print('Selected Outer Matrix',best_matrix)
P2_temp=np.matmul(data_intrinsics['K2'],best_matrix)
pts3d = sub.triangulate(P1, load_temple_coords['pts1'], P2_temp, a)

# 8. Figure out the correct P2
print('pts3d.shape',pts3d.shape)

# 9. Scatter plot the correct 3D points
fig = plt.figure()
res = fig.add_subplot(111, projection='3d')
res.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],c='b', marker='o')
res.set_xlabel('X')
res.set_ylabel('Y')
res.set_zlabel('Z')
#res.view_init(elev=10., azim=75)
plt.show()
out_P1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
R1=out_P1[:,0:3]
T1=out_P1[:,3:4]
R2=P2[:,:,0][:,0:3]
T2=P2[:,:,0][:,3:4]
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
np.savez('../data/extrinsics.npz', R1=R1, R2=R2, t1=T1, t2=T2)