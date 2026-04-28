import os
import sys
import numpy as np
# import logging
import cv2
from PIL import Image
import time, glob
from fractions import Fraction
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from Module_Utils import get_reference_facial_points, init_face_mask, warp_and_crop_face, parsmask, parsmaskv2, warp_and_crop_face_with_head, calculate_iou
from module.Inference.HPInference_FaceDetector import FaceDetector
from module.Inference.HPInference_FaceRestore import FaceRestore
import onnxruntime as ort
import argparse
import ast
class FaceEnhance():
    def __init__(self, modelname, detail=0):
        self.threshold = 0.9
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        default_square = True
        # self.reference_5pts = get_reference_facial_points((512, 512), inner_padding_factor, outer_padding, default_square)
        self.reference_5pts = np.array([[192.98138, 239.94708+50], [318.90277, 240.1936+50], [256.63416, 314.01935+50], [201.26117, 371.41043+50], [313.08905, 371.15118+50]])
        self.mask = init_face_mask()
        self.modelname = modelname
        self.model_FaceRestore = FaceRestore(modelname)
        self.model_FaceDetector = FaceDetector()
        onnx_path = 'model/faceparsing.onnx'
        self.parsemask = ort.InferenceSession(onnx_path, providers=['DmlExecutionProvider','CPUExecutionProvider'])
        self.parsing = True
        self.two = False
        self.detail = detail

    def normalize(self, image):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float16)
        image = (image/255. - mean) / std
        return image
    
    def maskface(self, tfm_inv, restored_face,om, ow, oh):
        if self.parsing:
            tfm_data = tfm_inv.reshape(-1)[:6]
            img = self.normalize(restored_face)
            img = np.transpose(img, [2, 0, 1])
            img = img.astype(np.float32)
            input_img = np.expand_dims(img, 0)                
            ort_inputs = {'input': input_img}
            ort_output = self.parsemask.run(None, ort_inputs)[0] 
            ort_output = np.squeeze(ort_output, 0)
            ort_output = ort_output.argmax(0)

            if 'zhuangshi' in self.modelname:
                tmp_mask = parsmaskv2(ort_output, thres=20, blurlen=5)
            else:
                tmp_mask = parsmask(ort_output)

            tmp_mask_img = Image.fromarray(tmp_mask, mode='L')
            tmp_mask_img = tmp_mask_img.resize(restored_face.shape[:2])
            tmp_mask_img = tmp_mask_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_mask = np.asarray(tmp_mask_img)
            tmp_mask = tmp_mask[:, :, np.newaxis]

            tmp_img = Image.fromarray(restored_face.astype(np.uint8))
            tmp_img = tmp_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_img = np.asarray(tmp_img)
            om = om * (1 - tmp_mask / 255.0) + tmp_img[:,:,::-1] * (tmp_mask / 255.0)                

        else:   
            tmp_mask = self.mask
            tmp_mask_img = Image.fromarray(tmp_mask, mode='L')
            tmp_mask_img = tmp_mask_img.resize(restored_face.shape[:2])
            tfm_data = tfm_inv.reshape(-1)[:6]
            tmp_mask_img = tmp_mask_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_mask = np.asarray(tmp_mask_img)

            tmp_mask = tmp_mask[:, :, np.newaxis]
            tmp_img = Image.fromarray(restored_face.astype(np.uint8))
            tmp_img = tmp_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_img = np.asarray(tmp_img)
            om = om * (1 - tmp_mask / 255.0) + tmp_img[:,:,::-1] * (tmp_mask / 255.0)        
        return om, tmp_mask
    
    def sharpe(self, of, restored_face):
        img1=np.array(of[:,:,::-1],dtype=np.float32)
        res1=img1-cv2.blur(img1,ksize=[5,5])
        img2=np.array(restored_face,dtype=np.float32)
        img3=cv2.blur(img2,ksize=[5,5])+res1*self.detail
        img3=np.clip(img3,0,255).astype(np.uint8)
        return img3

    def process_face(self, src, bg_img, name, face_rect=None):
        om = np.asarray(bg_img[:,:,::-1])
        src_img = Image.fromarray(src[:,:,::-1])
        oh, ow = om.shape[:2]
        t0 = time.time()
        # 人脸检测
        facebs, landms = self.model_FaceDetector.run(src)
        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            # 画人脸矩形框
            # npa_om = cv2.rectangle(np.array(om), (int(faceb[0]), int(faceb[1])), (int(faceb[2]), int(faceb[3])), (0, 255, 0), 2)
            # cv2.imwrite(name+'.png', npa_om)
            if faceb[4] < self.threshold:
                continue
            elif face_rect:
                src_box = faceb[:4]
                iou = calculate_iou(src_box, face_rect)
                print('iou--------',iou)
                if iou<0.6:
                    continue
            facial5points = np.reshape(facial5points, (2, 5))
            of, tfm_inv = warp_and_crop_face(src_img, facial5points, reference_pts=(self.reference_5pts), crop_size=(512, 512))
            restored_face = self.model_FaceRestore.run(of)
            if self.detail>1:
                restored_face = self.sharpe(of, restored_face)
            om, tmp_mask = self.maskface(tfm_inv, restored_face,om, ow, oh)

        om = om.astype(np.uint8)
        om = cv2.cvtColor(om, cv2.COLOR_RGB2BGR)
        return om

def process_files(input_paths, output_path, face_rect):
    for i, img_path in enumerate(input_paths):
        start_total_time = time.time()
        name = os.path.basename(img_path)
        suffix = name.split('.')[-1]
        output_img_path = os.path.join(output_path, name.split('.')[0]+'.png')
        nparr = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = face_enhancer.process_face(image, image, name, face_rect)
        cv2.imencode('.png', result)[1].tofile(output_img_path)
        end_total_time = time.time()
        print(f'图片 {name} {i+1}/{len(input_paths)} 处理完成，耗时: {end_total_time - start_total_time:.2f} 秒')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="appface妆容迁移")
    parser.add_argument("-t","--type", help="处理类型,0-1", type=int, default=0) # 0-女 1-男
    parser.add_argument("-d", "--detail", help="人脸细节保留程度1-1.5", type=float, default=1)
    parser.add_argument("-f", "--face_rect", help="指定人脸矩阵", type=str, default=None)
    args = parser.parse_args()

    if args.face_rect:
        face_rect = ast.literal_eval(args.face_rect)
    else:
        face_rect = None
    input_paths = glob.glob(os.path.join(r'inputs','*'))
    output_path = r'outputs'
    female_type = ['hlw_1','hlw_2','keai_3','sihua_4','zhuangshi_5','zhuangshi2_6','meili_7','reqing_8','youya_9','guangze_10','dianying_11','qingfu_12']
    male_type = ['hlw','hlw2','keai','sihua','meili','reqing','guangze','dianying','qingfu']
    if args.type==0:
        for typename in tqdm(female_type):
            modelname = os.path.join("model/appface_transfer_female",typename+'.onnx')
            face_enhancer = FaceEnhance(modelname, args.detail)
            if args.detail>1:
                output_type_path = os.path.join(output_path,'female_'+str(args.detail), typename)
            else:
                output_type_path = os.path.join(output_path,'female', typename)
            os.makedirs(output_type_path, exist_ok=True)
            process_files(input_paths, output_type_path, face_rect)     
    else:
        # typename = male_type[args.type]
        # print('typename---', typename)
        for typename in tqdm(male_type):
            modelname = os.path.join("model/appface_transfer_male",typename+'.onnx')
            face_enhancer = FaceEnhance(modelname, args.detail)
            if args.detail>1:
                output_type_path = os.path.join(output_path, 'male_'+str(args.detail), typename)
            else:
                output_type_path = os.path.join(output_path, 'male', typename) 
            os.makedirs(output_type_path, exist_ok=True)
            process_files(input_paths, output_type_path, face_rect)
