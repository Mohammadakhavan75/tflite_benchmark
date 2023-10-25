from tflite_loader import model_loader
import tflite_runtime.interpreter as tflite
import argparse
from data_loader import data_loader

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', help='File path of image file', type=str, required=True)
    parser.add_argument('--delegate_path', help='File path of ArmNN delegate file', type=str, defualt=None)
    parser.add_argument('--preferred_backends', help='list of backends in order of preference', defualt=None, type=str, nargs='+', required=False, default=["CpuAcc", "CpuRef"]
)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parsing()
    # delegate_path = args.delegate_path
    # backends = args.preferred_backends
    # backends = ",".join(backends)
    # #load the delegate
    # args.armnn_delegate = tflite.load_delegate(delegate_path,
    # options={
    #     "backends": backends,
    #     "logging-severity": "info"})
    
    loader = model_loader(args)
    data_loader = data_loader(args)
    img = data_loader.load()
    out = loader.inference(img)