import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import pickle
from scipy.spatial.distance import cdist
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
import cv2 as cv

# 读取图像
I = cv.imread('../data/landscape/sun_aciebeapldilzbug.jpg')
I_rgb = cv.cvtColor(I, cv.COLOR_BGR2RGB)  # 转换为RGB
# 读取图片对应的字典
dictionary = pickle.load(open('../data/landscape/sun_aciebeapldilzbug_Harris.pkl', 'rb'))
dictionary_1 = pickle.load(open('../data/landscape/sun_aciebeapldilzbug_Random.pkl', 'rb'))

# 将wordMap转换为彩色图像
wordMap_color = label2rgb(dictionary)
wordMap_color_1 = label2rgb(dictionary_1)

# 显示原始图像和wordMap
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(I_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wordMap_color)
plt.title('Visual Words_Harris')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wordMap_color_1)
plt.title('Visual Words_Random')
plt.axis('off')

plt.show()
