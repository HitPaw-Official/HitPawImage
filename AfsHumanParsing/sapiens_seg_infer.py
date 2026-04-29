# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm
from ultralytics import YOLO

# torchvision.disable_beta_transforms_warning()

# timings = {}
# BATCH_SIZE = 32
checkpoint = "/home/ubuntu02/liuji/projects/ckpts/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
USE_TORCHSCRIPT = '_torchscript' in checkpoint
# build the model from a checkpoint file
exp_model = torch.jit.load(checkpoint)
dtype = torch.float32  # TorchScript models use float32
exp_model = exp_model.to("cuda:0")

def _demo_mm_inputs(batch_size, input_shape):
    (C, H, W) = input_shape
    N = batch_size
    rng = np.random.RandomState(0)
    imgs = rng.rand(batch_size, C, H, W)
    if torch.cuda.is_available():
        imgs = torch.Tensor(imgs).cuda()
    return imgs


def warmup_model(model, batch_size):
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=model.dtype).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=model.dtype
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s

def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs, batch_size=1):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)

def img_save_and_viz(
    image, result, classes=GOLIATH_CLASSES, palette=GOLIATH_PALETTE, title=None, opacity=0.5, threshold=0.3, 
):
    # image = image.data.numpy() ## bgr image

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()


    num_classes = len(classes)
    sem_seg = pred_sem_seg
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    mask = np.zeros_like(image)
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_image = (image_rgb * (1 - opacity) + mask * opacity).astype(np.uint8)
    mask1 = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) == 0
    mask1 = np.expand_dims(mask1*1.0, axis=2)
    cv2.imwrite("mask.jpg", (mask1*255).astype(np.uint8))
    vis_image = (image_rgb * mask1 + vis_image * (1-mask1)).astype(np.uint8)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    # vis_image = np.concatenate([image, vis_image], axis=1)
    return vis_image

def img_save_and_viz_detect(
    ori_size, box, mask,  result,
    classes=GOLIATH_CLASSES, palette=GOLIATH_PALETTE, title=None, opacity=0.5, threshold=0.3):
    # image = image.data.numpy() ## bgr image
    x1, y1, x2, y2 = box
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=(y2-y1, x2-x1), mode="bilinear"
    ).squeeze(0)
    print("seg_logits.shape: ", seg_logits.shape)
    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)
    pred_sem_seg = pred_sem_seg.data[0].numpy()
    pred_sem_seg_full = np.zeros(ori_size, dtype=np.uint8)
    pred_sem_seg_full[y1:y2, x1:x2] = pred_sem_seg
    num_classes = len(classes)
    sem_seg = pred_sem_seg_full
    ids = np.unique(sem_seg)[::-1]
    ids = ids.tolist()
    ids.remove(0)
    ids = np.array(ids)
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    
    for label, color in zip(labels, colors):
        mask[sem_seg == label, :] = color

    return mask
   
def img2tensor(img, shape=[1024, 768], mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5]): 
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    if shape:
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = img[[2, 1, 0], ...].float()
    if mean is not None and std is not None:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        img = (img - mean) / std
    return img.unsqueeze(0)

def enlarge_box(x_min, y_min, x_max, y_max, w, h, rate_w=0.05, rate_h=0.05):
    enlarge_x, enlarge_h = int(rate_w*(x_max-x_min)), int(rate_h*(y_max-y_min))
    # ll = max(enlarge_h, enlarge_x)
    # enlarge_x, enlarge_h = ll, ll
    x_min = int(max(0, x_min-enlarge_x))
    x_max = int(min(w, x_max+enlarge_x))
    y_min = int(max(0, y_min-enlarge_h))
    y_max = int(min(h, y_max+enlarge_h))
    return x_min, y_min, x_max, y_max


class SegTorchScript():
    def __init__(self, checkpoint="/home/ubuntu02/liuji/projects/ckpts/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
                        dtype=torch.float32, device="cuda:0"):
        # build the model from a checkpoint file
        exp_model = torch.jit.load(checkpoint)  # TorchScript models use float32
        self.exp_model = exp_model.to(device)
        self.dtype = dtype
        self.detect_model = YOLO("/home/ubuntu02/liuji/projects/uniAnimate/checkpoints/yolov8x.pt", task='detect')


    def detect(self, img):
        h, w = img.shape[:2]
        results = self.detect_model(img, classes=[0], conf=0.4)
        boxes = results[0].boxes.data.cpu().numpy().tolist()
        res = []
        if len(boxes) == 0:
            print("No object detected !")
            res.append([0, 0, w, h])
        else:
            boxes.sort(key=lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse=True)
            res = []
            max_area = (boxes[0][2]-boxes[0][0])*(boxes[0][3]-boxes[0][1])
            if len(boxes) == 1:
                rate_w = 0.1
                rate_h = 0.1
            else:
                rate_w = 0.05
                rate_h = 0.05
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                if (x2-x1)*(y2-y1) / max_area > 0.3:
                    x1, y1, x2, y2 = enlarge_box(x1, y1, x2, y2, w, h, rate_w, rate_h)
                    res.append([x1, y1, x2, y2])
        return res
    
    def run_with_detect(self, img):
        ori_size = img.shape[:2]

        boxes_list = self.detect(img.copy())
        mask = np.zeros_like(img)
        for i, (x1, y1, x2, y2) in enumerate(boxes_list):
            img_cut = img[y1:y2, x1:x2, :]
            batch_imgs = img2tensor(img_cut)
            # batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
            result = inference_model(self.exp_model, batch_imgs, dtype=self.dtype)[0]
            mask = img_save_and_viz_detect(ori_size, (x1, y1, x2, y2), mask, result)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * (1 - 0.5) + mask * 0.5).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def run(self, img):
        batch_imgs = img2tensor(img.copy())
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(self.exp_model, batch_imgs, dtype=self.dtype)[0]
        img_vis = img_save_and_viz(img, result)
        return img_vis


if __name__ == "__main__":
    # sts = SegTorchScript()
    # img_path = "/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/3.png"
    # img_np = cv2.imread(img_path)
    # res = sts.run(img_np)
    # cv2.imwrite("/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/3_sapiens.jpg", res)
    # print("successful")
    checkpoint = "/home/ubuntu02/liuji/projects/ckpts/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    sts = SegTorchScript(checkpoint=checkpoint)
    img_dir = "/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/productor_images_sr"
    save_dir = "/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/productor_images_sr"
    os.makedirs(save_dir, exist_ok=True)
    for img_name in os.listdir(img_dir):

        img_path = os.path.join(img_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        img_np = cv2.imread(img_path)
        res = sts.run_with_detect(img_np)
        
        cv2.imwrite(save_path, res)
