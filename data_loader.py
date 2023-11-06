import cv2
import numpy as np
import tensorflow as tf
import time
import torchvision
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader
from pathlib import Path
import re
import yaml


# TODO: I have to make shape of image automatic
class data_loader():
    def __init__(self, img_shape, model_shape):
        self.img_shape = img_shape
        self.model_shape = model_shape
        self.ret = True
        

    def load_img(self, path):
        img_np = cv2.imread(path)
        img_np = cv2.resize(img_np, (self.img_shape, self.img_shape))
        if img_np.shape[0] != self.model_shape[1]:
            img_np = img_np.transpose(2, 0, 1)
        self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
        self.img_tf = tf.expand_dims(self.img_tf , axis=0)
        return self.img_tf
    

    def load_vid(self, path, model, log=False):
        self.img_list = []
        # Open the video file
        video_capture = cv2.VideoCapture(path)  # Replace 'your_video.mp4' with your video file's path
        
        # Check if the video file was successfully opened
        if not video_capture.isOpened():
            print("Error: Could not open the video file.")
            exit()

        # Loop through the video frames
        times = []
        while True:
            self.ret, frame = video_capture.read()  # Read a frame from the video

            # Check if the frame was successfully read
            if not self.ret:
               break  # Break the loop if the video has ended

            # Process and display the frame (e.g., you can show it using cv2.imshow)
            img_np = cv2.resize(frame, (self.img_shape, self.img_shape))
            if img_np.shape[0] != self.model_shape[1]:
                img_np = img_np.transpose(2, 0, 1)
            self.img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)
            self.img_tf = tf.expand_dims(self.img_tf , axis=0)
            
            s = time.time()
            out = model.inference(self.img_tf)
            e = time.time()
            times.append(e-s)
            if log:
                print(f"Inference time is: {e-s}")
            
        # Release the video capture object and close any open windows
        video_capture.release()
        return times

class yolo_data_loader():
    def __init__(self) -> None:
        pass


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)


def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

def get_dataloader(self, dataset_path, batch_size):
    """Construct and return dataloader."""
    dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
    return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader



def metric_calcualte(preds):
    for batch_i, batch in enumerate(bar):
        self.batch_i = batch_i
        self.update_metrics(preds, batch)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data['yaml_file'] = str(file)
        return data

def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices '
                           f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
        if isinstance(names[0], str) and names[0].startswith('n0'):  # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names


def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    data = yaml_load(dataset, append_filename=True)
    data['nc'] = len(data['names'])

    # Resolve paths
    path = Path(data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    data['path'] = path  # download scripts

    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path

    return data  # dictionary

