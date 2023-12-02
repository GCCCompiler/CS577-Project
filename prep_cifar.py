import numpy as np
import os
import pickle
import cv2
import matplotlib.pyplot as plt

def load_cifar_10_data(data_dir):
    train_images, train_labels = [], []

    # 循环加载
    for i in range(1, 6):
        data_path = os.path.join(data_dir, f'data_batch_{i}')
        with open(data_path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            images = data[b'data']
            labels = data[b'labels']
            images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
            train_images.append(images)
            train_labels.extend(labels)

    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels).reshape(-1, 1)

    # 加载测试集
    test_images, test_labels = [], []
    test_data_path = os.path.join(data_dir, 'test_batch')
    with open(test_data_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
        test_images = data[b'data']
        test_labels = data[b'labels']
        test_images = test_images.reshape((len(test_images), 3, 32, 32)).transpose(0, 2, 3, 1)

    test_labels = np.array(test_labels).reshape(-1, 1)

    return (train_images, train_labels), (test_images, test_labels)


def save_images(test_images, test_labels):
    for i in range(len(test_images)):
        label = test_labels[i][0]
        folder_name = f'figures_dataset/cifar/{label}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_name = f'{folder_name}/CIFAR_{i:05d}.JPG'
        plt.imsave(file_name, test_images[i])

# 指定本地CIFAR-10数据集的路径
cifar_10_data_dir = 'D:/python code/visual_prompting-main/figures_dataset/cifar-10-batches-py'

(train_images, train_labels), (test_images, test_labels) = load_cifar_10_data(cifar_10_data_dir)

# 标准化数据
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# 查看数据维数信息
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

save_images(test_images, test_labels)
