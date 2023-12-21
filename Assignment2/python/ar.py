import numpy as np
import cv2
#Import necessary functions
import skimage.io
import skimage.color
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
from planarH import compositeH
from loadVid import loadVid
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize

# 读取照片
ar_video_path = '../data/ar_source.mov'
book_vid_path = '../data/book.mov'

# 加载video
ar_e = loadVid(ar_video_path)
book_VID = loadVid(book_vid_path)
extend = book_VID.shape[0] - ar_e.shape[0]

# 创建0填充数组
pad = np.zeros((extend, ar_e.shape[1], ar_e.shape[2], ar_e.shape[3]))
for i in range(extend):
    pad[i, :, :, :] = ar_e[i, :, :, :]

# 填充与拼接
ar = np.concatenate((ar_e, pad), axis=0)

# 初始化数据
locs1_arr = []
locs2_arr = []
matches_arr = []
bestH2to1_arr = []
cv_cover = cv2.imread('../data/cv_cover.jpg')
composite_list = []

# 每一帧数据处理
for i in range(book_VID.shape[0]):
    print('i :{} and book_vid :{}'.format(i, book_VID[i, :, :, :].shape))
    matches, locs1, locs2 = matchPics(book_VID[i, :, :, :], cv_cover)
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    # 计算单应矩阵的RANSAC
    bestH2to1, inliers = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])
    dim = (cv_cover.shape[1], cv_cover.shape[0])

    # 更改形状
    aspect_ratio = cv_cover.shape[1] / cv_cover.shape[0]

    video_cover = ar[i, :, :, :]
    video_cover = video_cover[44:-44, :]

    # 尺寸修正-长宽比-高
    H, W, C = video_cover.shape
    h = H / 2
    w = W / 2
    width_ar = H * cv_cover.shape[1] / cv_cover.shape[0]
    video_cover = video_cover[:, int(w - width_ar / 2):int(w + width_ar / 2)]
    video_cover = cv2.resize(video_cover, dim)

    # 模板与输出图像叠加
    composite_img = compositeH(bestH2to1, video_cover, book_VID[i, :, :, :])
    # 添加数据
    composite_list.append(composite_img)

# 替换数据，另存
composite = composite_list
format = "XVID"
fps = 25
fourcc = VideoWriter_fourcc(*format)
output_vid_name = "../result/output_31.mov"
vid = None
size = None
for img in composite:
    if vid is None:
        if size is None:
            size = img.shape[1], img.shape[0]
        vid = VideoWriter(output_vid_name, fourcc, float(fps), size, True)
    vid.write(np.uint8(img))
vid.release()

#Write script for Q4.1
