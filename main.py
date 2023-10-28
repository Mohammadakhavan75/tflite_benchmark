from tflite_loader import model_loader
# import tflite_runtime.interpreter as mytflite
import argparse
import numpy as np
from data_loader import data_loader
import time 
import os
import cv2
import tensorflow as tf
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
    parser.add_argument('--vid_path', help='Path for video file', type=str, default=None)
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
    elif args.vid_path is not None:
        # imgs = loader.load_vid(args.vid_path)
        video_capture = cv2.VideoCapture(args.vid_path)
        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            exit()

        times=[]
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Break the loop if the video has ended

            img_np = cv2.resize(frame, (model.input_shape, model.input_shape))
            if img_np.shape[0] != model.model_shape[1]:
                img_np = img_np.transpose(2, 0, 1)
            img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
            img_tf = tf.expand_dims(img_tf , axis=0)
            s = time.time()
            out = model.inference(img_tf)
            e = time.time()
            times.append(e-s)
            # print(f" Inference time is: {e-s}")

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f" Average time is: {1 / np.mean(times)}")
                break

        # Release the video capture object and close any open windows
        video_capture.release()
        print(f" Average time is: {1 / np.mean(times)}")
        exit()
    else:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            exit()

        times=[]
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Break the loop if the video has ended

            img_np = cv2.resize(frame, (model.input_shape, model.input_shape))
            if img_np.shape[0] != model.model_shape[1]:
                img_np = img_np.transpose(2, 0, 1)
            img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
            img_tf = tf.expand_dims(img_tf , axis=0)
            s = time.time()
            out = model.inference(img_tf)
            e = time.time()
            times.append(e-s)
            print(f" Inference time is: {e-s}")

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f" Average time is: {1 / np.mean(times)}")
                break

        # Release the video capture object and close any open windows
        video_capture.release()
        print(f" Average time is: {1 / np.mean(times)}")
        exit()


    print("Data loaded")
    
    times = []
    for img in imgs:
        s = time.time()
        out = model.inference(img)
        e = time.time()
        times.append(e-s)
    
    print(f"output is recived {out} \n shape is: {out.shape},\
           argmax: {np.argmax(out, axis=1)}\n \
            average time is: {np.mean(times)} average frame rate is: {1 / np.mean(times)}")
