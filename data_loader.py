import cv2
import numpy as np
import tensorflow as tf

class data_loader():
    def __init__(self, img_path, img_shape):
        img_np = cv2.imread(img_path)
        img_np = cv2.resize(img_np, (img_shape, img_shape))
        print(f"######### {img_np.shape}")
        img_np = img_np.transpose(2, 0, 1)
        self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
        self.img_tf = tf.expand_dims(self.img_tf , axis=0)

    def load(self):
        return self.img_tf