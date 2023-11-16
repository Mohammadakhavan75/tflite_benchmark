from tflite_loader import model_loader
# import tflite_runtime.interpreter as mytflite
import argparse
import numpy as np
from data_loader import data_loader, yolo_data_loader
import time 
import os
from tqdm import tqdm
from utils import metrics, object_detection_metrics

"""
I should write a test for all possible inputs
for example when I change a thing I should run these both:

python main.py --img_path ../R.jpg --model_path ../yolov8n.tflite
python main.py --vid_path ../data0.avi --model_path ../yolov8n.tflite

So I make sure that everything is working properly

"""


def parsing():
    parser = argparse.ArgumentParser(description='',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', help='Path for image file', type=str, default=None)
    parser.add_argument('--img_folder', help='Path for image folder', type=str, default=None)
    parser.add_argument('--vid_path', help='Path for video file', type=str, default=None)
    parser.add_argument('--yaml_file', help='Path for video file', type=str, default=None)
    parser.add_argument('--mode', help='Mode can be classification|object detection|object tracking|lane detection|stream', type=str, default=None)
    parser.add_argument('--annot_type', help='Annotation type can be coco', type=str, default='coco')
    parser.add_argument('--model_path', help='Model path file', type=str, required=True)
    parser.add_argument('--device', help='Device can be cuda or cpu or None', type=str, default=None)
    parser.add_argument('--delegate_path', help='File path of ArmNN delegate file', type=str, default=None)
    parser.add_argument('--preferred_backends', help='list of backends in order of preference', type=str, nargs='+', required=False, default=["CpuAcc", "CpuRef"])
    args = parser.parse_args()
    args.armnn_delegate = None

    return args


if __name__ == '__main__':
    args = parsing()
    # delegate_path = args.delegate_path
    # backends = args.preferred_backends
    # backends = ",".join(backends)
    # #load the delegate
    # args.armnn_delegate = mytflite.load_delegate(delegate_path,
    # options={
    #     "backends": backends,
    #     "logging-severity": "info"})
    
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = model_loader(args)
    print("Model initiated")
    loader = data_loader(img_shape=model.input_shape, model_shape=model.model_shape)

    print("Data loader initiated")

    if args.img_path is not None:
        imgs = loader.load_img(args.img_path)
        times = []
        for img in imgs:
            s = time.time()
            out = model.inference(img)
            e = time.time()
            times.append(e-s)
        
        print(f"average time is: {np.mean(times)} average frame rate is: {1 / np.mean(times)}")

    elif args.img_folder and args.mode=="classification":
        img_folders = os.listdir(args.img_folder)
        img_folders_paths = []
        for img_folder in img_folders:
            img_folders_paths.append(os.path.join(args.img_folder, img_folder))
        
        imgs_full_path = []
        imgs_labels = []
        for i, path in enumerate(img_folders_paths):
            imgs_path = os.listdir(path)
            for img_path in imgs_path:
                imgs_full_path.append(os.path.join(path, img_path))
                imgs_labels.append(i)

        times = []
        acc = []
        preds = []
        for i, img in enumerate(tqdm(imgs_full_path)):
            img = loader.load_img(img)
            s = time.time()
            out = model.inference(img)
            e = time.time()
            times.append(e-s)
            acc_i = np.argmax(out, axis=1) == imgs_labels[i]
            preds.append(np.argmax(out, axis=1))
            acc.append(acc_i)

        metric = metrics(y_pred=preds, y_true=imgs_labels)

        print(f"avg acc: {np.mean(acc)}\n \
                average time is: {np.mean(times)}\n \
                    average frame rate is: {1 / np.mean(times)}")
        print(f"FPR: {metric.FPR}, TPR: {metric.TPR}, FNR: {metric.FNR}, TNR: {metric.TNR}")
        
    elif args.vid_path is not None:
        times = loader.load_vid(args.vid_path, model, log=True)
        print(f"Average inference time is: {np.mean(times)}\n \
              Average frame rate is: {1 / np.mean(times)}")
        exit()
    # Stream from webcam
    elif args.mode == 'stream': 
        times = loader.load_vid(0, model)
        print(f"Average inference time is: {np.mean(times)}\n \
              Average frame rate is: {1 / np.mean(times)}")
        exit()
    elif args.mode == 'object detection':
        yolo = yolo_data_loader(args.yaml_file)
        loader = yolo.load(type='val')
        obj_metrics = object_detection_metrics()
        for batch_i, batch in enumerate(loader):
            preds = model(batch['img'], augment=yolo.args)
            preds = obj_metrics.post_processing(preds)
            obj_metrics.update_metrics(preds, batch)

