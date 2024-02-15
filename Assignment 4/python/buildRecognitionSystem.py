import numpy as np
# -----fill in your implementation here --------
import pickle
import glob
import os

from createFilterBank import create_filterbank
from getImageFeatures import get_image_features

traintest_file = open('../data/traintest.pkl', 'rb')
traintest = pickle.load(traintest_file)
traintest_file.close()

# filterBank
filterBank = create_filterbank()

# check random
random_pkl = []
harris_pkl = []

# 聚类数量
K = 100

# 遍历训练集中的图像路径
for i, path in enumerate(traintest['train_imagenames']):
    # 打开存储Random提取特征的二进制文件，加载文件内容
    pkl_path = open('../data/%s_%s.pkl'%(path[:-4], 'Random'), 'rb')
    pkl = pickle.load(pkl_path)
    pkl_path.close()
    random_pkl.append(pkl)

    # 打开存储Harris提取特征的二进制文件，加载文件内容
    pkl_path = open('../data/%s_%s.pkl'%(path[:-4], 'Harris'), 'rb')
    pkl = pickle.load(pkl_path)
    pkl_path.close()
    harris_pkl.append(pkl)

# get trainlabels
trainLabels = traintest['train_labels']

# random dictionary
with open('dictionaryRandom.pkl', 'rb') as handle:
    random_dic = pickle.load(handle)

# train features
random_hist = get_image_features(random_pkl, K)

visionRandom = {'dictionary' : random_dic, 'filterBank' : filterBank, 'trainFeatures' : random_hist, 'trainLabels' : trainLabels}
with open('visionRandom.pkl', 'wb') as handle:
    pickle.dump(visionRandom, handle, protocol=pickle.HIGHEST_PROTOCOL)

# harris dictionary
with open('dictionaryHarris.pkl', 'rb') as handle:
    harris_dic = pickle.load(handle)

harris_hist = get_image_features(harris_pkl, K)

visionHarris = {'dictionary' : harris_dic, 'filterBank' : filterBank, 'trainFeatures' : harris_hist, 'trainLabels' : trainLabels}
with open('visionHarris.pkl', 'wb') as handle:
    pickle.dump(visionHarris, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------
