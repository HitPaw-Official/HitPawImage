"""
人脸修复模型Win端和Mac推理：
输入尺寸（1, 3, 512, 512）不限
输出和输入尺寸一致
"""

import os
import sys
import cv2
import numpy as np
import toml
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.Inference.Inference_Utils import init_model, normalize_image, name_in, name_out, leaky_relu


class FaceRestore(object):
    def __init__(self, modelname):
        # 从Inference_config.toml中导入模型初始化所需的参数
        self.config = toml.load("configs/Inference_config.toml")
        self.platform = self.config["modelinit"]["platform"]
        self.device = self.config["modelinit"]["device"]
        self.deviceid = self.config["modelinit"]["deviceid"]
        self.timerecord = None
        # if self.platform == 'macOS':
        #     modelname = self.config["FaceRestoreCoreml"]["modelname"]
        #     self.session = init_model(modelname, self.device, self.deviceid)
        # else:
        #     modelname = self.config["FaceRestoreOnnx"]["modelname"]
        self.session = init_model(modelname, self.device, self.deviceid)

    def preprocess(self, img):
        # 归一化
        img = (img - 127.5) / 127.5
        # 视频人脸修复模型的输入尺寸为256
        # img = cv2.resize(img, (256, 256))

        img = img.astype('float32')
        # H, W, C -> C, H, W
        img = img.transpose(2, 0, 1)
        # C, H, W -> B, C, H, W
        input = img[np.newaxis, :].transpose(0, 1, 2, 3)

        return input

    def post_processing(self, output):
        # C, H, W -> H, W, C
        img_cv = output.transpose(1, 2, 0)
        # 将输出的归一化值变为0～255
        np.clip(img_cv, -1.0, 1.0, img_cv)
        img_cv = (img_cv + 1) / 2
        img_cv = (img_cv * 255.0).round()
        # float32 转 uint8,
        restored_face = img_cv.astype(np.uint8)

        return restored_face

    def run(self, image):

        begin = time.time()
        #模型前处理
        input = self.preprocess(image)

        # 模型推理
        inferstart = time.time()

        if self.platform == 'Win':
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            out = self.session.run([output_name], {input_name: input})
            result = out[0].squeeze(0)
        elif self.platform == 'macOS':
            results = self.session.predict({self.config["FaceRestoreCoreml"]["inputname"]: input})
            result = list(results.values())[self.config["FaceRestoreCoreml"]["outputindex"]][0]

        inferend = time.time()
        #模型后处理
        restored_face = self.post_processing(result)

        restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)

        end = time.time()

        pre = (inferstart - begin) * 1000
        inf = (inferend - inferstart) * 1000
        pos = (end - inferend) * 1000

        self.timerecord = {"FR_pre_time": pre, "FR_inf_time": inf, "FR_pos_time": pos}

        return restored_face


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../")
    print(os.getcwd())
    dirp = r'/home/zhengshuting/project/SR/GFPGAN/results/bad50w/cropped_faces/1d2fabecb5ade1632d3cd9bb0bc7c1ae_320p+双人+小脸+遮挡_00.png'
    # save = r'\\192.168.1.7\测试部\AI底层\zhengst\单帧人脸生成\diffbir\结果\年龄\softmodel_faces'
    # os.makedirs(save, exist_ok=True)
    start = time.time()
    # for item in sorted(os.listdir(dir)[:20]):
        # path = os.path.join(dir, item)
    img = cv2.imread(dirp)
    # src = cv2.imdecode(np.fromfile(path, dtype=np.uint8),1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (512, 512))
    # print(path)

    face_enhancer = FaceRestore()

    dst_img = face_enhancer.run(img)

    # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
    end = time.time()
    cost = end - start
    print(cost/20)
    cv2.imwrite('/home/zhengshuting/project/Tool/Enhancer_inference/tmp_align.png', dst_img)
    # logging.info(f"FaceRestore cost time: %f s" % cost)
        # res = cv2.imencode('.png', dst_img)
        # print(len(res))
        # res[1].tofile(os.path.join(save, item))
        
