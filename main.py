from tflite_loader import model_loader
# import tflite_runtime.interpreter as mytflite
import argparse
import numpy as np

def parsing():
    parser = argparse.ArgumentParser(description='',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', help='File path of image file', type=str, required=True)
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
    
    loader = model_loader(args)
    out = loader.inference(args)
    print(f"output is recived {out} \n shape is: {out.shape}, argmax: {np.argmax(out, axis=1)}")