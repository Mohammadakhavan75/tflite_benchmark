import cv2
import numpy as np
import tensorflow as tf

class data_loader():
    def __init__(self, args):
        img_np = cv2.imread(args.img_path)
        self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

    def load(self):
        return self.img_tf