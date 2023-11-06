import cv2
import numpy as np
import tensorflow as tf
import time

from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader
from pathlib import Path
import re
import yaml
from ultralytics.utils import TQDM, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO


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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class yolo_data_loader():
    def __init__(self, yaml_path) -> None:
        self.yaml_path = yaml_path
        self.data = self.check_det_dataset(yaml_path)
        self.workers = 1
        self.args = Namespace(task='detect', mode='val', model='yolov8n.pt', data='coco128.yaml', epochs=100,\
                            patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None,\
                            workers=0, project=None, name=None, exist_ok=False, pretrained=True, optimizer='auto',\
                            verbose=True, seed=0, deterministic=True, single_cls=False, rect=True, cos_lr=False,\
                            close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None,\
                            overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split='val', save_json=False,\
                            save_hybrid=False, conf=0.001, iou=0.7, max_det=300, half=False, dnn=False, plots=True,\
                            source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True,\
                            show_conf=True, vid_stride=1, stream_buffer=False, line_width=None, visualize=False, augment=False,\
                            agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format='torchscript', keras=False,\
                            optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False,\
                            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,\
                            warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0,\
                            nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0,\
                            perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker='botsort.yaml')
    
    # Finilized
    def check_det_dataset(self, dataset):
        """
        verify dataset.

        This function checks the availability of a specified dataset, and if not found, it has the option to download and
        unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
        resolves paths related to the dataset.

        Returns:
            (dict): Parsed dataset information and paths.
        """

        data = self.yaml_load(dataset, append_filename=True)
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
    

    # Finilized
    def yaml_load(self, file='data.yaml', append_filename=False):
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


    # Finilized
    def check_class_names(self, names):
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
                names_map = self.yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']  # human-readable names
                names = {k: names_map[v] for k, v in names.items()}
        return names


    def build_yolo_dataset(self, cfg, img_path, batch, data, mode='train', rect=False, stride=32):
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
            return self.build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode)


    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        return build_dataloader(dataset, batch_size, self.workers, shuffle=False, rank=-1)  # return dataloader

 
    def load(self, batch_size, type='val'):
        return self.get_dataloader(self.data.get(type), batch_size)
    

