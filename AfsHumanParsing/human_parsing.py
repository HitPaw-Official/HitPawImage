import os
import cv2
import torch
import numpy as np
import copy
from PIL import Image
import torch.nn.functional as F
from classes_and_palettes import COMMON_GOLIATH_CLASSES, COLOR_DICT, COLOR_DICT_GRAY
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer



class RealESRGAN():
    def __init__(self, model_path, netscale=4, dni_weight=None, tile=0, tile_pad=10, pre_pad=0):
        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
                                scale=netscale,
                                model_path=model_path,
                                dni_weight=dni_weight,
                                model=model,
                                tile=tile,
                                tile_pad=tile_pad,
                                pre_pad=pre_pad,
                                half=True,
                                gpu_id=None)
        
    def run(self, img):
        output, _ = self.upsampler.enhance(img, outscale=4)
        return output
    
def resize_image(img, ref_size=800):
    # 获取原始宽度和高度
    height, width = img.shape[:2]
    
    # 计算缩放比例
    if width > height:
        scale = ref_size / width
    else:
        scale = ref_size / height
    
    # 计算新尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 保存调整后的图像
    return resized_img

def fake_pad_images_to_batchsize(imgs, batch_size=1):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)

def get_sapiens_result(
    image, result, classes=COMMON_GOLIATH_CLASSES, threshold=0.3
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
    parts = [classes[label] for label in labels]
    out_dict = {}
    for label, part in zip(labels, parts):
        if part != "Background":
            if part not in out_dict:
                out_dict[part] = sem_seg == label
            else:
                out_dict[part] = np.logical_or(out_dict[part], sem_seg == label)
    return out_dict


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

def do_human_parsing(result, h, w):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    out_dict = {}
    human_mask = np.zeros((h, w))
    for i in range(len(labels)):
        if labels[i] != "Human" and labels[i] != 'Background':
            if scores[i] > 0.3:
                lmask = cv2.resize(np.clip(masks[i], 0, 1), (w, h), cv2.INTER_NEAREST)
                out_dict[labels[i]] = lmask
                human_mask = np.logical_or(lmask, human_mask)
    return out_dict, human_mask

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

class HumanParsing():
    def __init__(self, sapiens_ckpt="ckpts/AfsHumanParsing/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
                    m2fp_ckpt="ckpts/AfsHumanParsing/cv_resnet101_image-multiple-human-parsing",
                    detect_ckpt="ckpts/AfsHumanParsing/yolov8x.pt",
                    real_esrgan_ckpt="ckpts/AfsHumanParsing/RealESRGAN_x4plus.pth",
                    dtype=torch.float32, device="cuda:0"):
        self.area_thre = 1024*768
        self.dtype = dtype
        # build the model from a checkpoint file
        sapiens_model = torch.jit.load(sapiens_ckpt)  # TorchScript models use float32
        self.sapiens_model = sapiens_model.to(device)
        
        # yolo detect
        self.detect_model = YOLO(detect_ckpt, task='detect')

        # m2fp
        self.segmentation_pipeline = pipeline(
                                            Tasks.image_segmentation,
                                            m2fp_ckpt,
                                            )

        # real esrgan
        self.real_esrgan = RealESRGAN(model_path=real_esrgan_ckpt)


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
        torch.cuda.empty_cache()
        return res
    
    def infer_real_esrgan(self, img):
        out = self.real_esrgan.run(img)
        torch.cuda.empty_cache()
        return out
    
    def infer_sapiens(self, img):
        h, w = img.shape[:2]
        is_upscale = False
        if h*w < self.area_thre:
            # super resolution
            is_upscale = True
            img = self.infer_real_esrgan(img)
        batch_imgs = img2tensor(img)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        with torch.no_grad():
            results = self.sapiens_model(batch_imgs.to(self.dtype).cuda())
        results = [r.cpu() for r in results][0]
        sapiens_dict = get_sapiens_result(img, results)
        for key, val in sapiens_dict.items():
            val = val.astype(np.uint8)
            if is_upscale:
                val = cv2.resize(val, (val.shape[1]//4, val.shape[0]//4), cv2.INTER_NEAREST)
            sapiens_dict[key] = val
        torch.cuda.empty_cache()
        return sapiens_dict
    
    def infer_m2fp(self, img):
        ori_h, ori_w = img.shape[:2]
        img_input = resize_image(img)
        img_input  = Image.fromarray(img_input)
        result = self.segmentation_pipeline(img_input)
        parsing_masks_dict, human_mask = do_human_parsing(result, ori_h, ori_w)
        torch.cuda.empty_cache()  
        return parsing_masks_dict, human_mask
    
    def combine_m2fp_sapiens(self, m2fp_d, sapiens_d):
        # 1.检查两者的完整性， 用解析多的
        # 2.头发用m2fp
        m2fp_count, sapiens_count = 0, 0
        for _, v in m2fp_d.items():
            m2fp_count += np.sum(v>0)
        for _, v in sapiens_d.items():
            sapiens_count += np.sum(v>0)

        if m2fp_count > 1.5*sapiens_count:
            return m2fp_d

        if "Hair" in m2fp_d:
            sapiens_d["Hair"] = m2fp_d["Hair"]
        if "Sunglasses" in m2fp_d:
            sapiens_d["Sunglasses"] = m2fp_d["Sunglasses"]
        if "Hat" in m2fp_d:
            sapiens_d["Hat"] = m2fp_d["Hat"]

        if "Face_Neck" in sapiens_d:
            if "Face" in m2fp_d:
                sapiens_d["Face"] = m2fp_d["Face"]
                if "Torso-skin" in m2fp_d:
                    neck_mask = np.logical_and(sapiens_d["Face_Neck"], m2fp_d["Torso-skin"]).astype(np.uint8)
                else:
                    neck_mask = np.logical_xor(sapiens_d["Face_Neck"], m2fp_d["Face"]).astype(np.uint8)
                    neck_mask = cv2.erode(neck_mask, np.ones((3,3),np.uint8), iterations=1)
                    neck_mask = cv2.dilate(neck_mask, np.ones((3,3),np.uint8), iterations=1)
                if "Torso-skin" in sapiens_d:
                    sapiens_d["Torso-skin"] = np.logical_or(sapiens_d["Torso-skin"], neck_mask)
                else:
                    if "Torso-skin" in m2fp_d:
                        sapiens_d["Torso-skin"] = m2fp_d["Torso-skin"]
                    else:
                        sapiens_d["Torso-skin"] = neck_mask
            else:
                print("Face not in m2fp")
            del sapiens_d["Face_Neck"]
        return sapiens_d

    def run_with_detect_app(self, img):
        ori_size = img.shape[:2]
        boxes_list = self.detect(img.copy()[:, :, ::-1])
        # m2fp_out = img.copy()
        # sapiens_out = img.copy()
        com_out = img.copy()
        # tmp_img = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes_list):
            img_cut = img[y1:y2, x1:x2, :]
            sapiens_res = self.infer_sapiens(img_cut.copy())
            m2fp_res, human_mask = self.infer_m2fp(img_cut.copy())
            human_mask = human_mask.astype(np.uint8)
            human_mask = cv2.dilate(human_mask, np.ones((21, 21), np.uint8), iterations=3)
            for k,v in sapiens_res.items():
                 sapiens_res[k] = np.logical_and(human_mask, v).astype(np.uint8)
            com_res = self.combine_m2fp_sapiens(copy.deepcopy(m2fp_res), copy.deepcopy(sapiens_res))

            for label_name, mask_cut in com_res.items():
                mask = np.zeros(ori_size)
                mask[y1:y2, x1:x2] = mask_cut
                com_out = draw_transparent_red(com_out, mask, COLOR_DICT[label_name])
                # tmp_out = draw_transparent_red(tmp_img, mask, COLOR_DICT[label_name])
                # cv2.imwrite(os.path.join("/home/ubuntu02/liuji/projects/M2FP/test_data/outputs/222", label_name+".jpg"), tmp_out)
        final_out = np.concatenate((img, com_out), axis=1)
        torch.cuda.empty_cache()
        return final_out
    
    def run_with_detect(self, img):
        ori_size = img.shape[:2]
        boxes_list = self.detect(img.copy()[:, :, ::-1])
        final_out = np.zeros(ori_size)
        for i, (x1, y1, x2, y2) in enumerate(boxes_list):
            img_cut = img[y1:y2, x1:x2, :]
            sapiens_res = self.infer_sapiens(img_cut.copy())
            m2fp_res, human_mask = self.infer_m2fp(img_cut.copy())
            human_mask = human_mask.astype(np.uint8)
            human_mask = cv2.dilate(human_mask, np.ones((21, 21), np.uint8), iterations=3)
            for k, v in sapiens_res.items():
                 sapiens_res[k] = np.logical_and(human_mask, v).astype(np.uint8)
            com_res = self.combine_m2fp_sapiens(copy.deepcopy(m2fp_res), copy.deepcopy(sapiens_res))

            for label_name, mask_cut in com_res.items():
                mask = np.zeros(ori_size)
                mask[y1:y2, x1:x2] = mask_cut
                gray_id = int(COLOR_DICT_GRAY[label_name])
                final_out[mask>0] = gray_id             
        torch.cuda.empty_cache()    
        return final_out  


if __name__ == '__main__':
    sapiens_ckpt = "/home/ubuntu02/liuji/projects/ckpts/AfsHumanParsing/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    m2fp_ckpt = "/home/ubuntu02/liuji/projects/ckpts/AfsHumanParsing/cv_resnet101_image-multiple-human-parsing"
    detect_ckpt = "/home/ubuntu02/liuji/projects/ckpts/AfsHumanParsing/yolov8x.pt"
    real_esrgan_ckpt = "/home/ubuntu02/liuji/projects/ckpts/AfsHumanParsing/RealESRGAN_x4plus.pth"
    hp = HumanParsing(sapiens_ckpt=sapiens_ckpt,
                      m2fp_ckpt=m2fp_ckpt,
                      detect_ckpt=detect_ckpt,
                      real_esrgan_ckpt=real_esrgan_ckpt,
                      )

    img_path = './test.jpg'
    # img_path = './test.jpg'
    img = cv2.imread(img_path)
    res = hp.run_with_detect(img)
    cv2.imwrite('./test_result.png', res)

