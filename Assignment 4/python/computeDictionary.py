import pickle
import os
from getDictionary import get_dictionary

# -----fill in your implementation here --------
def compute_dict(dictname, dict):
    with open(dictname, 'wb') as handle:
        pickle.dump(dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return 0

def get_image_paths(root_folder):
    valid_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.gif']
    image_paths = []

    for subdir, dirs, files in os.walk(root_folder):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                # 获取相对于根文件夹的相对路径
                relative_path = os.path.relpath(os.path.join(subdir, filename), root_folder)
                # 使用斜杠替换路径分隔符
                relative_path = relative_path.replace(os.path.sep, '/')
                image_paths.append(relative_path)

    return image_paths

# 文件夹路径
root_folder = '../data'
img_paths_dict = get_image_paths(root_folder)

def process_images_and_save_dictionary(imgPaths, alpha, K, method, save_path):
    # 处理图像并构建字典
    dictionary = get_dictionary(imgPaths, alpha, K, method)

    # 保存字典为pickle文件
    compute_dict(save_path, dictionary)

    print(f"字典已成功保存至 {save_path}")


folder_path = '../data'
img_paths = get_image_paths(folder_path)
print(img_paths)

# 指定参数
alpha = 50  # 随机点或哈里斯点的数量
K = 100  # K均值聚类的数量
method = 'Harris'  # 'Random' 或 'Harris'
save_path = 'Harris_dictionary.pkl'  # 保存的pickle文件路径

# 调用函数
process_images_and_save_dictionary(img_paths, alpha, K, method, save_path)

# ----------------------------------------------



