import numpy as np
import matplotlib.pyplot as plt
from submission import estimate_pose, estimate_params
# write your implementation here

def project_3d_to_2d(points_3d, camera_matrix):
    """
    将三维点投影到图像平面上的二维点
    """
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    projected_2d_homogeneous = camera_matrix @ points_3d_homogeneous.T
    projected_2d = (projected_2d_homogeneous[:2, :] / projected_2d_homogeneous[2, :]).T
    return projected_2d


def project_cad_model(cad_model_points, camera_matrix):
    """
    将CAD模型的三维点投影到图像平面上的二维点
    """
    cad_model_homogeneous = np.hstack((cad_model_points, np.ones((cad_model_points.shape[0], 1))))
    projected_cad_homogeneous = camera_matrix @ cad_model_homogeneous.T
    projected_cad = (projected_cad_homogeneous[:2, :] / projected_cad_homogeneous[2, :]).T
    return projected_cad


# 加载数据
data = np.load('../data/pnp.npz', allow_pickle=True)
image = data['image']
cad_model = data['cad'][0][0][0]  # Simplified access to CAD model
points_3d = data["X"]
points_2d = data['x']

# 估计姿态和参数
camera_matrix = estimate_pose(points_2d, points_3d)
intrinsic_matrix, rotation_matrix, translation_vector = estimate_params(camera_matrix)

# 投影三维点到二维
projected_2d_points = project_3d_to_2d(points_3d, camera_matrix)

# 设置绘图
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(image)
ax.scatter(points_2d[:, 0], points_2d[:, 1], c='purple', label='Given 2D Points', s=40)
ax.scatter(projected_2d_points[:, 0], projected_2d_points[:, 1], facecolors='none', edgecolors='yellow', label='Projected CAD Points', s=100)
ax.set_title('2D and Projected 3D Points')
ax.legend()

# 绘制三维CAD模型
ax3d = fig.add_subplot(132, projection='3d')
ax3d.scatter(cad_model[:, 0], cad_model[:, 1], cad_model[:, 2], edgecolors='yellow')

# 绘制投影的CAD模型顶点
ax2d = fig.add_subplot(133)
ax2d.imshow(image, cmap='gray')
projected_cad_points = project_cad_model(cad_model, camera_matrix)
ax2d.scatter(projected_cad_points[:, 0], projected_cad_points[:, 1], c='yellow')
plt.show()
