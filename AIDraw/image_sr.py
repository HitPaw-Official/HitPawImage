import os
import argparse
import math
import torch

from PIL import Image
from sr.sr_net import RRDBNet
from sr import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='checkpoint/v1.3')
    parser.add_argument('--images', type=list, default=['data/outputs/2023_5_4/t2i/0_test/1536_2048_1/1_1.jpg',
                                                        'data/outputs/2023_5_4/t2i/0_test/1536_2048_1/2_1.jpg',
                                                        'data/outputs/2023_5_4/t2i/0_test/1536_2048_1/3_1.jpg',
                                                        'data/outputs/2023_5_4/t2i/0_test/1536_2048_1/4_1.jpg',
                                                        'data/outputs/2023_5_4/t2i/0_test/1536_2048_1/5_1.jpg'])
    parser.add_argument('--sr_rate', type=int, default=4, choices=[2, 4])

    opt = parser.parse_args()

    return opt


class SR(object):
    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.sr2 = self._load_sr_model(2)

    def _load_sr_model(self, scale):
        sr_model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)
        if scale == 2:
            model_path = os.path.join(self.model_path, 'BSRGAN2.pth')
        else:
            model_path = os.path.join(self.model_path, 'BSRGAN4.pth')

        sr_model.load_state_dict(torch.load(model_path), strict=True)
        sr_model.eval()
        for k, v in sr_model.named_parameters():
            v.requires_grad = False

        return sr_model

    def infer(self, images, scale, device):
        hq_images = []
        if scale == 2:
            sr_model = self.sr2.to(device)
        # else:
        #     sr_model = self.sr4.to(device)

        for img in images:
            out_name = img.split('.')[0] + '_hq.' + img.split('.')[-1]
            img_L = utils.imread_uint(img, n_channels=3)
            img = utils.uint2tensor4(img_L)
            img = img.to(device)
            img_E = self.pred(img, sr_model, scale)
            Image.fromarray(img_E).save(out_name)
            hq_images.append(os.path.abspath(out_name))

        return hq_images

    @staticmethod
    def pred(img, model, scale, tile_size=128, tile_pad=10):
        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                import torch as ori_torch
                with ori_torch.no_grad():
                    output_tile = model(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
                img_E = utils.tensor2uint(output)

        return img_E

    def run(self, images, sr_rate):
        ori_images, hq_images = [], []
        # for b in images:
        if sr_rate == 4:
            hq_image_name = self.infer(images, 2, self.device)
            hq_image_list = []
            for i in hq_image_name:
                hq_image = Image.open(i)
                ori_w, ori_h = hq_image.size
                hq_image = hq_image.resize((ori_w * 2, ori_h * 2), resample=Image.Resampling.LANCZOS)
                hq_image.save(i)
                hq_image_list += [i]
        else:
            hq_image_list = self.infer(images, sr_rate, self.device)
        hq_images += hq_image_list
        ori_images += images

        torch.cuda.empty_cache()

        return ori_images, hq_images


def main():
    opt = parse_args()
    sr = SR('checkpoint/v1.3', opt.device)
    ori_images, hq_images = sr.run(opt.images, opt.sr_rate)
    print(ori_images)
    print(hq_images)


if __name__ == '__main__':
    main()
