from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import os

class COCOParser:
    def __init__(self, labels_root_path, imgs_root_path):
        self.imgs_root_path = imgs_root_path
        self.labels_root_path = labels_root_path
        self.labels_dict = {}
        self.imgs_dict = {}

    def load_labels(self):
        for filename in os.listdir(self.labels_root_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.labels_root_path, filename)
                
                with open(file_path, 'r') as file:
                    labels = file.read()
                    self.labels_dict[filename.split('.')[0]] = labels.split('\n')[:-1]

        return self.labels_dict
    
    def load_img_paths(self):
        for filename in os.listdir(self.imgs_root_path):
            if filename.endswith('.jpg'):
                file_path = os.path.join(self.imgs_root_path, filename)
                self.imgs_dict[filename] = file_path
        return self.imgs_dict
    
class metrics():
    def __init__(self, y_true, y_pred):
        self.c_m = confusion_matrix(y_true, y_pred)
        self.FP = self.c_m.sum(axis=0) - np.diag(self.c_m)  
        self.FN = self.c_m.sum(axis=1) - np.diag(self.c_m)
        self.TP = np.diag(self.c_m)
        self.TN = self.c_m.values.sum() - (self.FP + self.FN + self.TP)

    # Fall out or false positive rate
    def FPR(self):
        return self.FP / (self.FP + self.TN)
    
    # Sensitivity, hit rate, recall, or true positive rate    
    def TPR(self):
        return self.TP / (self.TP + self.FN)
    
    # Specificity or true negative rate
    def TNR(self):
        return self.TN / (self.TN + self.FP)

    def FNR(self):
        return self.FN / (self.TP + self.FN)
    
    # Precision or positive predictive value
    def Precision(self):
        return self.TP / (self.TP + self.FP)
    
    # Negative predictive value    
    def NPV(self):
        return self.TN / (self.TN + self.FN)
    
    # False discovery rate    
    def FDR(self):
        return self.FP / (self.TP + self.FP)
