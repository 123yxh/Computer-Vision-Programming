import numpy as np
from utils import chi2dist
from scipy.spatial.distance import euclidean

def get_image_distance(hist1, hist2, method):
    dist = []
    hist1 = hist1.flatten()  # 确保 hist1 是一维数组
    for trainhist in hist2:
        trainhist = trainhist.flatten()  # 确保 trainhist 也是一维数组
        if method == 'euclidean':
            dist_ = euclidean(hist1, trainhist)
        elif method == 'chi2':
            dist_ = chi2dist(hist1, trainhist)
        dist.append(dist_)
    return dist
