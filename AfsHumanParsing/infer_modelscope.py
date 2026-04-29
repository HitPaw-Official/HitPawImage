import os
import cv2
from PIL import Image
import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from ultralytics import YOLO


COLOR_DICT = {'Left-arm':(255, 0, 0), 'Skirt':(0, 255, 0), 'Hair':(0, 0, 255), 'Pants':(255, 255, 0), 'Sunglasses':(0, 255, 255),
              'Left-leg':(255, 0, 255), 'Torso-skin':(220, 220, 220), 'Face':(139, 0, 0),
              'UpperClothes':(45, 135, 15), 'Right-leg':(0, 0, 139), 'Right-arm':(0, 100, 0), 'Human':(20, 20, 20),
              'Coat':(255, 165, 0), 'Left-shoe':(255, 192, 203), 'Right-shoe':(128, 0, 128), 'Hat':(128, 128, 128), 'Dress':(144, 144, 0),
              'Socks':(255, 215, 0), 'Scarf':(144, 238, 144), 'Gloves':(148, 0, 211), 'Glove':(148, 0, 200)}
modelscope_ckpts_dir='/media/data1/projects/aigc/facechain/ckpts/modelscope/'
segmentation_pipeline = pipeline(
    Tasks.image_segmentation,
    'damo/cv_resnet101_image-multiple-human-parsing', #'damo/cv_resnet101_image-single-human-parsing', # 'damo/cv_resnet101_image-multiple-human-parsing',
    ) 


detect_model = YOLO("/home/ubuntu02/liuji/projects/uniAnimate/checkpoints/yolov8x.pt", task='detect')

def enlarge_box(x_min, y_min, x_max, y_max, w, h, rate_w=0.05, rate_h=0.05):
    enlarge_x, enlarge_h = int(rate_w*(x_max-x_min)), int(rate_h*(y_max-y_min))
    # ll = max(enlarge_h, enlarge_x)
    # enlarge_x, enlarge_h = ll, ll
    x_min = int(max(0, x_min-enlarge_x))
    x_max = int(min(w, x_max+enlarge_x))
    y_min = int(max(0, y_min-enlarge_h))
    y_max = int(min(h, y_max+enlarge_h))
    return x_min, y_min, x_max, y_max

def detect(img):
    h, w = img.shape[:2]
    results = detect_model(img, classes=[0], conf=0.4)
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

def draw_transparent_red(image, mask, color=(0, 0, 200)):
    # 确保掩码图像和原始图像具有相同的大小
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建一个透明红色的副本
    transparent_red = np.zeros_like(image, dtype=np.uint8)
    transparent_red[:, :] = color
    transparent_red[np.where(mask == 0)] = (0, 0, 0)

    # 将透明红色图像与原始图像叠加
    out = cv2.add(image, transparent_red)

    return out

def get_mask_face(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    img_shape = masks[0].shape
    mask_face = np.zeros(img_shape)

    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
    mask_face = np.clip(mask_face, 0, 1)
    mask_face = np.expand_dims(mask_face, 2)
    return mask_face

def do_human_parsing(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    print(labels)
    out_dict = {}
    for i in range(len(labels)):
        if labels[i] != "Human" and labels[i] != 'Background':
            if scores[i] > 0.3:
                out_dict[labels[i]] = np.clip(masks[i], 0, 1)
    return out_dict

def main(img):
    boxes_list = detect(img)
    print("boxes len: ", len(boxes_list))
    h, w = img.shape[:2]
    out = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_list):
        img_cut = img[y1:y2, x1:x2, :]
        # cv2.imwrite('/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/tmp_{}.jpg'.format(i), img_cut)
        img_cut = Image.fromarray(img_cut).convert("RGB")
        result = segmentation_pipeline(img_cut)
        parsing_masks_dict = do_human_parsing(result)
        
        for label_name, mask_cut in parsing_masks_dict.items():
            if label_name != 'Human':
                mask = np.zeros((h, w))
                mask[y1:y2, x1:x2] = mask_cut
                out = draw_transparent_red(out, mask, COLOR_DICT[label_name])
    return out

if __name__ == "__main__":
    # img_dir = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/productor_images'
    # save_dir = '/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/productor_images_modelscope'
    # os.makedirs(save_dir, exist_ok=True)
    # for name in os.listdir(img_dir):
    #     img_path = os.path.join(img_dir, name)
    #     # img_path = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/2.jpg'
    #     img = cv2.imread(img_path)
    #     res = main(img)
    #     cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), res.astype(np.uint8))

    # save_dir = '/home/ubuntu02/liuji/projects/M2FP/test_data/outputs'
    # os.makedirs(save_dir, exist_ok=True)
    # img_path = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/2.jpg'
    # img = cv2.imread(img_path)
    # res = main(img)
    # cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), res.astype(np.uint8))

    input_img = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/productor_images/1.jpg'
    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    result = segmentation_pipeline(input_img)
    print(len(result))
    parsing_masks_dict = do_human_parsing(result)
    img = cv2.imread(input_img)
    for label_name, mask in parsing_masks_dict.items():
        img_tmp = draw_transparent_red(img, mask, COLOR_DICT[label_name])
        cv2.imwrite('/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/111/'+label_name+'.jpg', img_tmp)