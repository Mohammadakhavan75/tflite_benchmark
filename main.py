from tflite_loader import model_loader
# import tflite_runtime.interpreter as mytflite
import argparse
import numpy as np
from data_loader import data_loader

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
    
    model = model_loader(args)
    loader = data_loader(img_shape=model.input_shape)
    if args.img_path is not None:
        imgs = loader.load_img(args.img_path)
    elif args.vid_path is not None:
        imgs = loader.load_vid(args.vid_path)
    else:
        # Stream
        pass

    for img in imgs:
        out = model.inference(img)

    print(f"output is recived {out} \n shape is: {out.shape}, argmax: {np.argmax(out, axis=1)}")

