import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    n = 0
    # 遍历wordMap每一行
    for wordmap in wordMap:
        # 使用 np.histogram 函数计算每行中单词的直方图
        hist = np.histogram(wordmap, bins = np.arange(dictionarySize))
        # 如果是第一行，初始化特征向量 h
        if n == 0:
            h = hist[0]/np.sum(hist[0])
        else:
            # 将每行的直方图标准化，并垂直堆叠到特征向量 h 中
            h = np.vstack((h, hist[0]/np.sum(hist[0])))
        n = n+1
    
    return h
