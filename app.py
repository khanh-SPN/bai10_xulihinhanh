import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


# Đường dẫn thư mục ảnh
train_dir = 'Train'
val_dir = 'Validation'


# Tiền xử lý dữ liệu
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(['Cat', 'Dog']):
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            if img is not None:
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


X_train, y_train = load_images_from_folder(train_dir)
X_val, y_val = load_images_from_folder(val_dir)


# Chuẩn hóa dữ liệu
X_train = X_train / 255.0
X_val = X_val / 255.0


# Chuyển đổi nhãn thành mảng one-hot cho ANN và CNN
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=2)


# 1. Mô hình SVM
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)


svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)
y_pred_svm = svm_model.predict(X_val_flat)
print("SVM Accuracy:", accuracy_score(y_val, y_pred_svm))


# 2. Mô hình ANN
ann_model = models.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])


ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train_onehot, epochs=10, validation_data=(X_val, y_val_onehot))


# 3. Mô hình CNN
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])


cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train_onehot, epochs=10, validation_data=(X_val, y_val_onehot))


# Dự đoán và in độ chính xác cho các mô hình
