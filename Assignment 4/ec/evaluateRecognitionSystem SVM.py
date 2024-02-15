import pickle
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
traintest_file = open('../data/traintest.pkl', 'rb')
traintest = pickle.load(traintest_file)
traintest_file.close()

with open('../python/dictionaryRandom.pkl', 'rb') as handle:
    dict_random = pickle.load(handle)

with open('../python/dictionaryHarris.pkl', 'rb') as handle:
    dict_harris = pickle.load(handle)

with open('../python/visionRandom.pkl', 'rb') as handle:
    train_random_histset = pickle.load(handle)

with open('../python/visionHarris.pkl', 'rb') as handle:
    train_harris_histset = pickle.load(handle)

# 使用线性核的 SVM 分类器
# svm_classifier_random = SVC(kernel='linear')
# svm_classifier_harris = SVC(kernel='linear')

# 使用多项式核的 SVM 分类器
svm_classifier_random = SVC(kernel='poly', degree=3)  # 可以调整 degree, gamma, coef0
svm_classifier_harris = SVC(kernel='poly', degree=3)

# 使用 RBF 核的 SVM 分类器
# svm_classifier_random = SVC(kernel='rbf')
# svm_classifier_harris = SVC(kernel='rbf')


# 使用训练集数据训练分类器
svm_classifier_random.fit(train_random_histset['trainFeatures'], train_random_histset['trainLabels'])
svm_classifier_harris.fit(train_harris_histset['trainFeatures'], train_harris_histset['trainLabels'])

K = 100
rand_eu_acc = []
rand_chi_acc = []
harris_eu_acc = []
harris_chi_acc = []

rand_svm_label = []
harris_svm_label = []

for i, path in enumerate(traintest['test_imagenames']):
    rand_wordmap_path = open('../data/%s_%s.pkl' % (path[:-4], 'Random'), 'rb')
    rand_wordmap = pickle.load(rand_wordmap_path)
    rand_wordmap_path.close()

    harris_wordmap_path = open('../data/%s_%s.pkl' % (path[:-4], 'Harris'), 'rb')
    harris_wordmap = pickle.load(harris_wordmap_path)
    harris_wordmap_path.close()

    rand_hist_ = np.histogram(rand_wordmap, bins=np.arange(K))
    rand_hist = rand_hist_[0] / np.sum(rand_hist_[0])
    rand_hist = rand_hist.reshape(1, -1)  # 为 SVM 准备

    harris_hist_ = np.histogram(harris_wordmap, bins=np.arange(K))
    harris_hist = harris_hist_[0] / np.sum(harris_hist_[0])
    harris_hist = harris_hist.reshape(1, -1)

    # 使用 SVM 分类器进行预测
    rand_label_pred = svm_classifier_random.predict(rand_hist)
    rand_svm_label.append(rand_label_pred[0])

    harris_label_pred = svm_classifier_harris.predict(harris_hist)
    harris_svm_label.append(harris_label_pred[0])

    # 计算准确率
    test_label = traintest['test_labels'][i]

    # 保存不同检测方式下的准确率
    rand_eu_result = 1 if (test_label == rand_label_pred[0]) else 0
    harris_eu_result = 1 if (test_label == harris_label_pred[0]) else 0

    rand_eu_acc.append(rand_eu_result)
    harris_eu_acc.append(harris_eu_result)

# 保存 SVM 分类结果
labelresult = {'random_svm_label': rand_svm_label, 'harris_svm_label': harris_svm_label}
with open('visionSVM.pkl', 'wb') as handle:
    pickle.dump(labelresult, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 计算混淆矩阵和准确率
rand_eu_mat = np.zeros((8, 8))
harris_eu_mat = np.zeros((8, 8))

for i in range(len(traintest['test_imagenames'])):
    gt = int(traintest['test_labels'][i])  # 真实标签
    rand_predict = int(rand_svm_label[i])
    harris_predict = int(harris_svm_label[i])

    rand_eu_mat[gt-1, rand_predict-1] += 1
    harris_eu_mat[gt-1, harris_predict-1] += 1

# 输出混淆矩阵和准确率
print('Random SVM Classification')
print('Confusion matrix:')
print(rand_eu_mat)

print('Harris SVM Classification')
print('Confusion matrix:')
print(harris_eu_mat)
print('Random SVM Classification_Accuracy_Polynomial nucleus: %f' % (sum(rand_eu_acc) / len(rand_eu_acc)))
print('Harris SVM Classification Accuracy_Polynomial nucleus: %f' % (sum(harris_eu_acc) / len(harris_eu_acc)))

# 设置绘图样式
sns.set(style='white')
# 创建绘图
plt.figure(figsize=(12, 6))
# 绘制 Random SVM 混淆矩阵
plt.subplot(1, 2, 1)
sns.heatmap(rand_eu_mat.astype(int), annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Random SVM Confusion Matrix_Polynomial nucleus')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 绘制 Harris SVM 混淆矩阵
plt.subplot(1, 2, 2)
sns.heatmap(harris_eu_mat.astype(int), annot=True, fmt='d', cmap='Reds', square=True)
plt.title('Harris SVM Confusion Matrix_Polynomial nucleus')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 显示图形
plt.tight_layout()
plt.show()
