import os
import shutil
import gradio as gr
import cv2
import datetime, time
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from ultralytics import YOLO


model = AutoModelForImageClassification.from_pretrained("/data/liuji/projects/nsfw_detection/run/nsfw_finetune_1e-4_224/checkpoint-495")
processor = ViTImageProcessor.from_pretrained('/data/liuji/projects/nsfw_detection/run/nsfw_finetune_1e-4_224/checkpoint-495')
detect_model = YOLO('yolo11x.pt')

def person_detect(img):
    imh, imw = img.shape[:2]
    results = detect_model(img.copy(), classes=[0], conf=0.5)
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

def get_nsfw_class(img):
    imgs_cut_list = person_detect(img)
    if len(imgs_cut_list) == 0:
        imgs_cut_list.append(img)

    predicted_max_label = 0
    for img_cut in imgs_cut_list:
        img_pil = Image.fromarray(img_cut[:,:,::-1])
        with torch.no_grad():
            inputs = processor(images=img_pil, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_max_label = max(logits.argmax(-1).item(), predicted_max_label)
    return model.config.id2label[predicted_max_label]


def main(files):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join('./demo_asset', now_str)
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, 'results.txt')
    writer = open(txt_path, 'w')
    img_paths = [x.name for x in files]
    img_paths = sorted(img_paths, key=lambda x:os.path.basename(x))
    for idx, file_path in enumerate(img_paths):
        try:
            name = os.path.basename(file_path)
            img = cv2.imread(file_path)
            nsfw_label = get_nsfw_class(img)
            writer.write("{} >>> {} \n".format(name, nsfw_label))
        except:
            continue
    writer.close()
    return txt_path


if __name__ == '__main__':
    demo = gr.Interface(
                        main,
                        gr.File(file_count="multiple", file_types=["image"]),
                        "file",
                        cache_examples=True
                        )
    demo.launch(server_name='0.0.0.0', server_port=7867, show_error=True, max_threads=1)