import tflite_loader
import tflite_runtime.interpreter as tflite
import argparse 

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_image', help='File path of image file', type=str, required=True)
    parser.add_argument('--delegate_path', help='File path of ArmNN delegate file', type=str, defualt=None)
    parser.add_argument('--preferred_backends', help='list of backends in order of preference', defualt=None, type=str, nargs='+', required=False, default=["CpuAcc", "CpuRef"]
)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parsing()
    model_path = ''
    armnn_delegate = tflite.load_delegate(args.delegate_path,options={
    "backends": backends,
    "logging-severity": "info"
    })
    tflite_loader()