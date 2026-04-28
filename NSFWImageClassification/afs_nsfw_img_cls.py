import cv2
import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForImageClassification, ViTImageProcessor
from ultralytics import YOLO
import time


# model = AutoModelForImageClassification.from_pretrained("/data/liuji/projects/nsfw_detection/run/nsfw_finetune_1e-4_224/checkpoint-495")
# processor = ViTImageProcessor.from_pretrained('/data/liuji/projects/nsfw_detection/run/nsfw_finetune_1e-4_224/checkpoint-495')
# detect_model = YOLO('yolo11x.pt')


class NsfwImageClassify():
    def __init__(self, nsfw_ckpt=None, detection_ckpt=None):
        self.model = AutoModelForImageClassification.from_pretrained(nsfw_ckpt)
        self.processor = ViTImageProcessor.from_pretrained(nsfw_ckpt)
        self.detect_model = YOLO(detection_ckpt)
        self.warmup()

    def warmup(self):
        for i in range(2):
            self.get_nsfw_class("warmup.png")
        
    def person_detect(self, img):
        imh, imw = img.shape[:2]
        results = self.detect_model(img.copy(), classes=[0], conf=0.5)
        boxes = results[0].boxes.xyxy
        imgs_list = []
        for box in boxes:
            box = box.cpu().numpy()
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            eng_x, eng_y = int(width*0.15), int(height*0.1)
            x1 = int(max(x1 - eng_x, 0))
            y1 = int(max(y1 - eng_y, 0))
            x2 = int(min(x2 + eng_x, imw))
            y2 = int(min(y2 + eng_y, imh))
            imgs_list.append(img[y1:y2, x1:x2, :])
        return imgs_list

    def get_nsfw_class(self, img_path):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)[:,:,::-1]
        imgs_cut_list = self.person_detect(img_np)
        if len(imgs_cut_list) == 0:
            imgs_cut_list.append(img_np)

        predicted_max_label = 0
        for img_cut in imgs_cut_list:
            img_pil = Image.fromarray(img_cut[:,:,::-1])
            with torch.no_grad():
                inputs = self.processor(images=img_pil, return_tensors="pt")
                outputs =self.model(**inputs)
                logits = outputs.logits
                predicted_max_label = max(logits.argmax(-1).item(), predicted_max_label)
        return self.model.config.id2label[predicted_max_label]
    

if __name__ == '__main__':
    # 初始化NSFW图像分类实例
    nic = NsfwImageClassify(nsfw_ckpt="/data/liuji/projects/nsfw_detection/run/nsfw_finetune_1e-4_224/checkpoint-495",
                            detection_ckpt='/data/liuji/projects/nsfw_detection/yolo11x.pt')
    img_path = "test.jpg"
    # 图像路径输入
    # 类别名称输出 [low, mid, high]
    out = nic.get_nsfw_class(img_path)
    print(out)
