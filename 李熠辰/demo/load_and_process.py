import pandas as pd
import cv2
import numpy as np


dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        #获取所有像素值，'pixels' 列包含了每张图片的像素值（以空格分隔的字符串）
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            # 将像素值从字符串转换为整数列表
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            # 将一维的像素列表转换为二维数组，重塑为 48x48 的图像
            face = np.asarray(face).reshape(width, height)
            # 调整图像大小为目标大小 (48, 48)，确保一致性
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x