from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torch
import torchvision

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


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    # t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def update_metrics(preds, batch):
    for si, pred in enumerate(preds):
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx]
        bbox = batch['bboxes'][idx]
        nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
        shape = batch['ori_shape'][si]
        correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
        self.seen += 1

        if npr == 0:
            if nl:
                if self.args.plots:
                    self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
            continue

        # Predictions
        if self.args.single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                        ratio_pad=batch['ratio_pad'][si])  # native-space pred

        # Evaluate
        if nl:
            height, width = batch['img'].shape[2:]
            tbox = xywh2xyxy(bbox) * torch.tensor(
                (width, height, width, height), device=self.device)  # target boxes
            scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space labels
            labelsn = torch.cat((cls, tbox), 1)  # native-space labels
            correct_bboxes = self._process_batch(predn, labelsn)
            # TODO: maybe remove these `self.` arguments as they already are member variable
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, labelsn)
