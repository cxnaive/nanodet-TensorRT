from abc import ABCMeta, abstractmethod
import random
import torch
import cv2
from ..transform.mosaic import merge_bboxes
import numpy as np
from torch.utils.data import Dataset
from ..transform import Pipeline


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.

    :param img_path: image data folder
    :param ann_path: annotation file path or folder
    :param use_instance_mask: load instance segmentation data
    :param use_seg_mask: load semantic segmentation data
    :param use_keypoint: load pose keypoint data
    :param load_mosaic: using mosaic data augmentation from yolov4
    :param mode: train or val or test
    """

    def __init__(self,
                 img_path,
                 ann_path,
                 input_size,
                 pipeline,
                 keep_ratio=True,
                 use_instance_mask=False,
                 use_seg_mask=False,
                 use_keypoint=False,
                 load_mosaic=False,
                 mixup=False,
                 mode='train'
                 ):
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.mixup = mixup
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)
        self.indices = list(range(len(self.data_info)))
        # print(self.indices)

    def __len__(self):
        return len(self.data_info)

    def get_valid_train_data(self, idx):
        while True:
            data = self.get_train_data(idx)
            if data is None:
                idx = self.get_another_id()
                continue
            else:
                return data

    def get_mosaic(self, idx):
        data = self.get_valid_train_data(idx)
        w, h = self.input_size
        # print(self.input_size)
        min_offset = 0.2
        cut_x = np.random.randint(
            int(w*min_offset), int(w*(1 - min_offset)))
        cut_y = np.random.randint(
            int(h*min_offset), int(h*(1 - min_offset)))
        data0 = data
        random_index = random.sample(self.indices, 3)
        data1 = self.get_valid_train_data(random_index[0])
        data2 = self.get_valid_train_data(random_index[1])
        data3 = self.get_valid_train_data(random_index[2])
        d1 = data0['img'][:, :cut_y, :cut_x]
        d2 = data1['img'][:, cut_y:, :cut_x]
        d3 = data2['img'][:, cut_y:, cut_x:]
        d4 = data3['img'][:, :cut_y, cut_x:]

        tmp1 = torch.cat([d1, d2], 1)
        tmp2 = torch.cat([d4, d3], 1)
        mosaic_img = torch.cat([tmp1, tmp2], 2)

        labels, boxes = merge_bboxes([data0['gt_bboxes'], data1['gt_bboxes'], data2['gt_bboxes'], data3['gt_bboxes']], [
            data0['gt_labels'], data1['gt_labels'], data2['gt_labels'], data3['gt_labels']], cut_x, cut_y, w, h)
        # print(data['gt_labels'])
        # print('before:',data['gt_labels'].dtype)
        data['gt_labels'] = np.array(labels, dtype=np.int64)
        data['gt_bboxes'] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        data['img'] = mosaic_img
        return data

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test':
            return self.get_val_data(idx)
        else:
            if self.load_mosaic:
                data = self.get_mosaic(idx)
            else:
                data = self.get_valid_train_data(idx)
            
            if self.mixup and random.random() < 0.5:
                if self.load_mosaic:
                    data1 = self.get_mosaic(self.get_another_id())
                else:
                    data1 = self.get_valid_train_data(self.get_another_id())

                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                data['img'] = data['img'] * r + data1['img'] * (1 - r)
                data['gt_labels'] = np.concatenate([data['gt_labels'],data1['gt_labels']])
                data['gt_bboxes'] = np.concatenate([data['gt_bboxes'],data1['gt_bboxes']])

            return data

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info)-1)
