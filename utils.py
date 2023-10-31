from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import confusion_matrix

class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        

    def load_id(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

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
