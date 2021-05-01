import os
import torch
import argparse
import numpy as np
import cv2
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar

from nanodet.util import mkdir, Logger, cfg, load_config, convert_old_model
from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.trainer.task import TrainingTask
from nanodet.evaluator import build_evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError('cfg.model.arch.head.num_classes must equal len(cfg.class_names),but got {} and {}'.format(cfg.model.arch.head.num_classes,len(cfg.class_names)))
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)

    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        pl.seed_everything(args.seed)

    logger.log('Setting up data...')
    # print(cfg.data.train)
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    evaluator = build_evaluator(cfg, val_dataset)

    show_cnt = 0
    for meta in train_dataset:
        img = meta['img'].numpy().transpose(1,2,0).copy()
        #cv2.imshow('gray',np.ones(img.shape,dtype=np.float32) * 0.5)
        #print(meta['gt_bboxes'])
        #print(img.shape)
        for idx in range(len(meta['gt_bboxes'])):
            rect = meta['gt_bboxes'][idx]
            #print(rect[0])
            label = meta['gt_labels'][idx]
            #print(label)
            cv2.rectangle(img, (int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),(0, 255, 0), 1)
            cv2.putText(img,str(cfg.class_names[label]),(int(rect[0]),int(rect[1] + 1.5)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
        cv2.imshow('dataset_img',img)
        cv2.waitKey(0)
        
        show_cnt += 1
        if show_cnt > 50:
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)