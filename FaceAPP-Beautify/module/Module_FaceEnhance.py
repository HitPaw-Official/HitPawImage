"""
人脸修复模块：
输入：原图
输出：彩图，尺寸经过放大
逻辑：先全图超分降噪，然后人脸检测，最后对每一个人脸做增强
"""

import os
import sys
import numpy as np
import logging
import cv2
from PIL import Image
import time
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module_Utils import get_reference_facial_points, init_face_mask, warp_and_crop_face
from Inference.HPInference_FaceDetector import FaceDetector
from Inference.HPInference_FaceRestore import FaceRestore
# from Inference.HPInference_topazSR import SR
from Module_Denoise import DenoiseAndSR
# from Inference.ori import TopazcropSR

class FaceEnhance():
    def __init__(self):
        self.threshold = 0.9
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        default_square = True

        # self.reference_5pts = get_reference_facial_points((512, 512), inner_padding_factor, outer_padding, default_square)
        self.reference_5pts = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],[201.26117, 371.41043], [313.08905, 371.15118]])
        self.mask = init_face_mask()

        self.model_FaceRestore = FaceRestore()
        self.model_FaceDetector = FaceDetector()
        # self.model_generalSR = DenoiseAndSR()
        # self.model_generalSR = TopazcropSR()
        # self.timelist = []

    def process(self, image):

        src = cv2.imdecode(np.fromfile(image,dtype=np.uint8),1)
        h,w = src.shape[:2]
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        #通用超分，将图片尺寸扩大4倍，并降噪
        # bg = self.model_generalSR.process(image)
        # cv2.imwrite(r'E:\data\face\video\datasets\avcface_safmn_org_26_512.png', bg)
        # print('ok-----------')
        # self.timelist.append(self.model_generalSR.timelist)
        bg_img = Image.fromarray(src)
        # bg_img = cv2.resize(src,(1280,1280))
        om = np.asarray(bg_img)
        print('src.shape-------------------',src.shape)
        src_img = Image.fromarray(src)
        # in_img = src_img.copy()


        oh, ow = om.shape[:2]
        # ow = oh = 1280
        # src_img = src_img.resize((ow, oh), Image.BICUBIC)
        # src_arr = np.asarray(src_img)
        t0 = time.time()
        # 人脸检测
        facebs, landms = self.model_FaceDetector.run(src)
        '''
        # facebs_np = np.array(facebs)
        # facebs = facebs[np.argmax(facebs_np[:,-1])].astype(np.uint)
        # of = src_arr[facebs[1]:facebs[3], facebs[0]:facebs[2]]
        # print('----------------',src_arr.shape, of.shape)
        # print('facebs------',facebs)
        '''
        # t1 = time.time()
        # # 人脸检测耗时统计存入FD_list列表
        # FD_list = []
        # FD_list.append(self.model_FaceDetector.timerecord)
        # # self.timelist.append(FD_list)

        # # 输出图像人脸数目
        # faces_num = len(facebs)
        # # print("The face number is: %d" % faces_num)
        # facebs_np = np.array(facebs)
        # if faces_num>=1:
        #     facial5points = landms[np.argmax(facebs_np[:,-1])]
        #     facial5points = np.reshape(facial5points, (2, 5))
        #     of, tfm_inv = warp_and_crop_face(src_img, facial5points, reference_pts=(self.reference_5pts), crop_size=(512, 512))
        # else:
        #     of = None
        #
        # FR_list = []

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
            else:
                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = warp_and_crop_face(src_img, facial5points, reference_pts=(self.reference_5pts), crop_size=(512, 512))
                ## tmp_p = facial5points / 4

                ## of, tfm_inv1 = warp_and_crop_face(in_img, tmp_p, reference_pts=(self.reference_5pts), crop_size=(512, 512))
                ## of1, tfm_inv = warp_and_crop_face(src_img, facial5points, reference_pts=(self.reference_5pts), crop_size=(512, 512))
                # 人脸进行修复
                # t2 = time.time()
                # cv2.imwrite('tmp_mask.png',of)
                # cv2.imwrite('tmp_maskof1.png',of1)
                # print('of----', of.shape)
                restored_face = self.model_FaceRestore.run(of)
                # print('3-----------')
                # t3 = time.time()
                # FR_list.append(self.model_FaceRestore.timerecord)

            tmp_mask = self.mask
            tmp_mask_img = Image.fromarray(tmp_mask, mode='L')
            tmp_mask_img = tmp_mask_img.resize(restored_face.shape[:2])
            tfm_data = tfm_inv.reshape(-1)[:6]
            tmp_mask_img = tmp_mask_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_mask = np.asarray(tmp_mask_img)

            tmp_mask = tmp_mask[:, :, np.newaxis]
            # cv2.imwrite('tmp_mask.png',tmp_mask)
            tmp_img = Image.fromarray(restored_face.astype(np.uint8))
            tmp_img = tmp_img.transform((ow, oh), method=(Image.AFFINE), data=tfm_data, resample=(Image.BILINEAR))
            tmp_img = np.asarray(tmp_img)
            # cv2.imwrite('tmp_img.png',tmp_img)

            om = om * (1 - tmp_mask / 255.0) + tmp_img * (tmp_mask / 255.0)

        # # self.timelist.append(FR_list)

        om = om.astype(np.uint8)
        om = cv2.cvtColor(om, cv2.COLOR_RGB2BGR)
        # # print('de--',t1-t0)
        # # print('wap--',t2-t1)
        # # print('gfpgan--',t3-t2)
        return om

if __name__ == '__main__':
    imgpath = r'D:\code'
    savepath = r'F:\code\faceclassonnx_infer\result'
    # self.paths_VF = sorted(glob.glob(os.path.join(imgpath,'*','*')))#[:200000]
    face_enhancer = FaceEnhance()#'./model/HP_FACE_ENHANCEMENT')
    tmp = None
    for itemfile in tqdm(sorted(os.listdir(imgpath))):
        os.makedirs(savepath, exist_ok=True)
        # os.makedirs(os.path.join(savepath,itemfile), exist_ok=True)
        # print('900:1000---')
        if itemfile != 'iris3_adist_1w_b10':
            continue
        for item in tqdm(sorted(os.listdir(os.path.join(imgpath, itemfile)))):
            # imgpath_item = os.path.join(imgpath, itemfile, item)
            imgpath_item = r"f:\code\faceclassonnx_infer\20241219153609.png"
            print(imgpath_item)
            dst_img = face_enhancer.process(imgpath_item)
            if dst_img is None:
                dst_img = tmp
            else:
                dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
                tmp = dst_img
            # cv2.imwrite(os.path.join(savepath,item),dst_img)
            cv2.imwrite(r"f:\code\faceclassonnx_infer\r.png",dst_img)
            break