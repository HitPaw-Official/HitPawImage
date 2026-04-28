"""
人脸检测模型Mac端推理：
输入尺寸（batchsize, channel, height, width）：1， 3， 512， 512

人脸检测模型Win端推理：
输入尺寸（1, 3, height, width）不限
"""

import os
import sys
import time
import numpy as np
import cv2
import toml

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from module.Inference.Inference_Utils import init_model, py_cpu_nms, cal_ratio, onnx_decode, onnx_decode_landm
from module.Inference.Inference_Utils import OnnxPriorBox

class FaceDetector(object):
    def __init__(self):
        # 从Inference_config.toml中导入模型初始化所需的参数
        self.config = toml.load("configs/Inference_config.toml")
        self.platform = self.config["modelinit"]["platform"]
        self.device = self.config["modelinit"]["device"]
        self.deviceid = self.config["modelinit"]["deviceid"]
        self.cfg_re50 = self.config["cfg_re50"]
        self.timerecord= None

        if self.platform == 'macOS':
            modelname = self.config["FaceDetectorCoreml"]["modelname"]
            self.session = init_model(modelname, self.device, self.deviceid)

        else:
            modelname = self.config["FaceDetectorOnnx"]["modelname"]
            self.session = init_model(modelname, self.device, self.deviceid)

    def run(self, image):
        begin = time.time()

        if self.platform == 'macOS':
            block_size_x = 512  # Mac 512
            block_size_y = 512  # Mac 512
        else:
            block_size_x = 1280 # Mac 512
            block_size_y = 1280 # Mac 512

        im0 = image

        h0, w0, _ = im0.shape
        ratio = cal_ratio(w0, h0, block_size_x, block_size_y)

        if ratio < 1:
            w1 = int(w0 * ratio)
            h1 = int(h0 * ratio)
            im_img = Image.fromarray(image)
            im_img = im_img.resize([w1, h1], resample=(Image.BICUBIC))
            image = np.asarray(im_img)
        else:
            w1, h1 = w0, h0

        image = np.pad(image, ((0, block_size_y - h1), (0, block_size_x - w1), (0, 0)), 'constant')

        resize = (ratio if ratio < 1 else 1)
        x_off = 0
        y_off = 0
        confidence_threshold = 0.9
        nms_threshold = 0.4
        top_k = 5000
        keep_top_k = 750

        im_height, im_width = image.shape[:2]
        src_arr = image.astype(np.float32) - (104, 117, 123)
        src_arr = np.expand_dims(src_arr.transpose(2, 0, 1), 0).astype(np.float32)

        scale = np.array([im_width, im_height, im_width, im_height])
        offset = np.array([x_off, y_off, x_off, y_off])

        inferstart = time.time()

        if self.platform == 'macOS':
            result = self.session.predict({self.config["FaceDetectorCoreml"]["inputname"]: src_arr})
            out = list(result.values())
            loc, conf, landms = out[0], out[2], out[1]
        else:
            input_name = self.session.get_inputs()[0].name
            out_name0 = self.session.get_outputs()[0].name
            out_name1 = self.session.get_outputs()[1].name
            out_name2 = self.session.get_outputs()[2].name
            out = self.session.run([out_name0, out_name1, out_name2], {input_name: src_arr})
            loc, conf, landms = out[0], out[1], out[2]

        inferend = time.time()

        priorbox = OnnxPriorBox(self.cfg_re50, image_size=(im_height, im_width))
        prior_data = priorbox.forward()
        boxes = onnx_decode(loc[0], prior_data, self.cfg_re50['variance'])
        boxes = boxes * scale / resize + offset
        scores = conf[0][:, 1]
        landms = onnx_decode_landm(landms[0], prior_data, self.cfg_re50['variance'])
        scale1 = np.array([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
        offset1 = np.array([x_off, y_off, x_off, y_off, x_off, y_off, x_off, y_off, x_off, y_off])
        landms = landms * scale1 / resize + offset1
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype((np.float32), copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10)

        end = time.time()

        pre = inferstart - begin
        inf = inferend - inferstart
        pos = end - inferend

        self.timerecord = {"FD_pre_time": pre, "FD_inf_time": inf, "FD_pos_time": pos}

        return (dets, landms)


if __name__ == '__main__':
    import time
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../")
    print(os.getcwd())

    img = cv2.imread(r"f:\code\faceclassonnx_infer\20241219153609.png")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, (512, 512))

    # width, height, _ = img.shape
    #
    # ow = width * 2
    # oh = height * 2
    #
    # src_img = Image.fromarray(img)
    # src_img = src_img.resize((ow, oh), Image.BICUBIC)
    # src_arr = np.asarray(src_img)

    start = time.time()
    model_FaceDetector = FaceDetector()
    mid = time.time()
    print(f"The Model Init cost : {(mid - start) * 1000} ms")

    dets, output = model_FaceDetector.run(img)

    end = time.time()
    print(f"The Inference cost : {(end - mid) * 1000} ms")

    faces_num = len(dets)

    print("end")


