import cv2
import os, time
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

#
class RealESRGAN():
    def __init__(self, model_name='RealESRGAN_x4plus_anime_6B', netscale=4, dni_weight=None, tile=0, tile_pad=10, pre_pad=0):
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = "weights/RealESRGAN_x4plus.pth"
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
        else:
            raise Exception("Sorry, no such model")
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
        self.face_enhancer = GFPGANer(
            model_path='gfpgan/weights/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)

    def run(self, img, use_enhancer=True):
        if use_enhancer:
            _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
           output, _ = self.upsampler.enhance(img, outscale=4)
        return output


resrg = RealESRGAN(model_name='RealESRGAN_x4plus') #  or model_name == 'RealESRGAN_x4plus'

# img_path = "/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/111/1.jpg"
# img = cv2.imread(img_path)
# t0 = time.time()
# out = resrg.run(img, use_enhancer=False)  # RealESRGAN_x4plus_anime_6B 不需要 face enhancer
# print("using time: ", time.time()-t0)
# cv2.imwrite("/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/111/1_sr.jpg", out)

img_dir = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/productor_images'
save_dir = '/home/ubuntu02/liuji/projects/M2FP/test_data/inputs/productor_images_sr'
os.makedirs(save_dir, exist_ok=True)
for name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, name)
    img = cv2.imread(img_path)
    if min(img.shape[:2]) > 1080:
        continue
    t0 = time.time()
    out = resrg.run(img, use_enhancer=False)  # RealESRGAN_x4plus_anime_6B 不需要 face enhancer
    print("using time: ", time.time()-t0)
    cv2.imwrite(os.path.join(save_dir, name), out)





