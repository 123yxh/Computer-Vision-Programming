import numpy as np

# -----fill in your implementation here --------
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from createFilterBank import create_filterbank
from getImageDistance import get_image_distance

# 加载测试集
traintest_file = open('../data/traintest.pkl', 'rb')
traintest = pickle.load(traintest_file)
traintest_file.close()

test_imagenames = traintest['test_imagenames']

# 获取滤波器响应
filterBank = create_filterbank()

# 加载数据.pkl
with open('dictionaryRandom.pkl', 'rb') as handle:
    dict_random = pickle.load(handle)

with open('dictionaryHarris.pkl', 'rb') as handle:
    dict_harris = pickle.load(handle)

with open('visionRandom.pkl', 'rb') as handle:
    train_random_histset = pickle.load(handle)

with open('visionHarris.pkl', 'rb') as handle:
    train_harris_histset = pickle.load(handle)

# 聚类数量
K = 100

# 初始化变量
rand_eu_acc = []
rand_chi_acc = []
harris_eu_acc = []
harris_chi_acc = []

rand_eu_label = []
rand_chi_label = []
harris_eu_label = []
harris_chi_label = []

for i, path in enumerate(traintest['test_imagenames']):
    # 打开Random字典的单词映射 pickle 文件并加载
    rand_wordmap_path = open('../data/%s_%s.pkl'%(path[:-4], 'Random'), 'rb')
    rand_wordmap = pickle.load(rand_wordmap_path)
    rand_wordmap_path.close()

    # 打开Harris字典的单词映射 pickle 文件并加载
    harris_wordmap_path = open('../data/%s_%s.pkl'%(path[:-4], 'Harris'), 'rb')
    harris_wordmap = pickle.load(harris_wordmap_path)
    harris_wordmap_path.close()

    # 计算Random单词映射的直方图并归一化
    rand_hist_ = np.histogram(rand_wordmap, bins = np.arange(K))
    rand_hist = rand_hist_[0]/np.sum(rand_hist_[0])
    rand_hist = rand_hist.reshape(-1, 1)
    # 计算Harris单词映射的直方图并归一化
    harris_hist_ = np.histogram(harris_wordmap, bins = np.arange(K))
    harris_hist = harris_hist_[0]/np.sum(harris_hist_[0])
    harris_hist = harris_hist.reshape(-1, 1)

    # 计算Random单词映射与训练数据之间的欧氏距离
    rand_dist_eu = get_image_distance(rand_hist, train_random_histset['trainFeatures'], 'euclidean')
    rand_eu_min_idx = np.argmin(rand_dist_eu)
    rand_eu_label.append(train_random_histset['trainLabels'][rand_eu_min_idx])

    # 计算Random单词映射与训练数据之间的卡方距离
    rand_dist_chi = get_image_distance(rand_hist, train_random_histset['trainFeatures'], 'chi2')
    rand_chi_min_idx = np.argmin(rand_dist_chi)
    rand_chi_label.append(train_random_histset['trainLabels'][rand_chi_min_idx])

    # 计算Harris单词映射与训练数据之间的欧式距离
    harris_dist_eu = get_image_distance(harris_hist, train_harris_histset['trainFeatures'], 'euclidean')
    # 返回距离测试集图像的索引返回
    harris_eu_min_idx = np.argmin(harris_dist_eu)
    # 保存标签
    harris_eu_label.append(train_harris_histset['trainLabels'][harris_eu_min_idx])

    # 计算Harris单词映射与训练数据之间的卡方距离
    harris_dist_chi = get_image_distance(harris_hist, train_harris_histset['trainFeatures'], 'chi2')
    harris_chi_min_idx = np.argmin(harris_dist_chi)
    harris_chi_label.append(train_harris_histset['trainLabels'][harris_chi_min_idx])

labelresult = {'random_eu_label' : rand_eu_label, 'random_chi_label' : rand_chi_label, 'harris_eu_label' : harris_eu_label, 'harris_chi_label' : harris_chi_label}
with open('nn_labelresult.pkl', 'wb') as handle:
    pickle.dump(labelresult, handle, protocol = pickle.HIGHEST_PROTOCOL)

for n in range(len(traintest['test_imagenames'])):
    test_label = traintest['test_labels'][n]

    # 如果测试的标签与真实标签相同，则输出1，反之
    rand_eu_result = 1 if (test_label == rand_eu_label[n]) else 0
    rand_chi_result = 1 if (test_label == rand_chi_label[n]) else 0

    harris_eu_result = 1 if (test_label == harris_eu_label[n]) else 0
    harris_chi_result = 1 if (test_label == harris_chi_label[n]) else 0

    # 保存对应的准确率
    rand_eu_acc.append(rand_eu_result)
    rand_chi_acc.append(rand_chi_result)
    harris_eu_acc.append(harris_eu_result)
    harris_chi_acc.append(harris_chi_result)

acc_result = {'rand_eu_acc' : rand_eu_acc, 'rand_chi_acc' : rand_chi_acc, 'harris_eu_acc' : harris_eu_acc, 'harris_chi_acc' : harris_chi_acc}
with open('AccResult.pkl', 'wb') as handle:
    pickle.dump(acc_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 设置绘图样式
sns.set(style='white')
# 创建绘图
plt.figure(figsize=(12, 6))

rand_eu_mat = np.zeros((8,8))
for i in range(len(traintest['test_imagenames'])):
    gt = int(traintest['test_labels'][i])  # 从 traintest['test_labels'] 中获取标签
    predict = int(rand_eu_label[i])
    rand_eu_mat[gt-1, predict-1] += 1
print('random euclidean metric')
print('confusion matrix : ')
print(rand_eu_mat)
print('random points & euclidean Accuracy : %f'%(sum(rand_eu_acc)/len(rand_eu_acc)))


rand_chi_mat = np.zeros((8, 8))
for i in range(len(traintest['test_imagenames'])):
    gt = int(traintest['test_labels'][i])
    predict = int(rand_chi_label[i])
    rand_chi_mat[gt - 1, predict - 1] += 1
print('[random chi metric]')
print('confusion matrix : ')
print(rand_chi_mat)
print('random points & chi Accuracy : %f' % (sum(rand_chi_acc) / len(rand_chi_acc)))


harris_eu_mat = np.zeros((8, 8))
for i in range(len(traintest['test_imagenames'])):
    gt = int(traintest['test_labels'][i])
    predict = int(harris_eu_label[i])
    harris_eu_mat[gt - 1, predict - 1] += 1
print('[harris euclidean metric]')
print('confusion matrix : ')
print(harris_eu_mat)
print('harris points & euclidean Accuracy : %f' % (sum(harris_eu_acc) / len(harris_eu_acc)))
plt.subplot(1, 2, 1)
sns.heatmap(harris_eu_mat.astype(int), annot=True, fmt='d', cmap='Blues', square=True)
plt.title('harris Confusion Matrix euclidean')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')


harris_chi_mat = np.zeros((8, 8))
for i in range(len(traintest['test_imagenames'])):
    gt = int(traintest['test_labels'][i])
    predict = int(harris_chi_label[i])
    harris_chi_mat[gt - 1, predict - 1] += 1
print('[harris chi metric]')
print('confusion matrix : ')
print(harris_chi_mat)
print('harris points & chi Accuracy : %f' % (sum(harris_chi_acc) / len(harris_chi_acc)))

plt.subplot(1, 2, 2)
sns.heatmap(harris_chi_mat.astype(int), annot=True, fmt='d', cmap='Reds', square=True)
plt.title('harris Confusion Matrix chi')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# 显示图形
plt.tight_layout()
plt.show()

# ----------------------------------------------
