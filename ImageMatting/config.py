"""
Rethinking Portrait Matting with Privacy Preserving
config file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""


########## Root paths and logging files paths


######### Paths of train datasets
TRAIN_PATHS_LIST=[
    "/home/liuji/datasets/matting_cutout/portrate/afs_matting_photoroom_train/",
    "/home/liuji/datasets/matting_cutout/portrate/PhotoMatte85/",
    "/home/liuji/datasets/matting_cutout/portrate/PPM-100/", 
    "/home/liuji/datasets/matting_cutout/portrate/train/",
    "/home/liuji/datasets/matting_cutout/portrate/VideoMatte240K_to_ImageMatte/",
    "/home/liuji/datasets/matting_cutout/portrate/portrait/",
    "/home/liuji/datasets/matting_cutout/portrate/human_half/",
    "/home/liuji/datasets/matting_cutout/portrate/ModanetDataset_photoroom_train/",
    "/home/liuji/datasets/matting_cutout/portrate/hard01/",
    "/home/liuji/datasets/matting_cutout/portrate/deepfashion_05/"
    ]


######### Paths of val datsets

VAL_PATHS_LIST = [
    "/home/liuji/datasets/matting_cutout/portrate/P3M-500-NP/",
    "/home/liuji/datasets/matting_cutout/portrate/ModanetDataset_photoroom_val/"
    ]



######### animal
# Paths of train datasets
ANIMAL_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting_cutout/animal/animal/",
                "/home/liuji/datasets/matting_cutout/animal/train/",
                "/home/liuji/datasets/matting_cutout/animal/AM-2K_compose01/",
                "/home/liuji/datasets/matting_cutout/animal/train01/",
                "/home/liuji/datasets/matting_cutout/portrate/train/"
    ]
# Paths of val datsets
ANIMAL_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting_cutout/animal/validation/"
    ]

######### car
# Paths of train datasets
CAR_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting_cutout/car/train01/"
    ]
# Paths of val datsets
CAR_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting_cutout/car/val01/"
    ]

######### clothes
# Paths of train datasets
CLOTHES_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting_cutout/clothes/train01"
    ]
# Paths of val datsets
CLOTHES_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting_cutout/clothes/val01"
    ]

######### food
# Paths of train datasets
FOOD_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting_cutout/food/train/"
    ]
# Paths of val datsets
FOOD_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting_cutout/food/val/"
    ]

######### furniture
# Paths of train datasets
FURNITURE_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting_cutout/furniture/train/"
    ]
# Paths of val datsets
FURNITURE_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting_cutout/furniture/val/"
    ]

######### sod
# Paths of train datasets
SOD_TRAIN_PATHS_LIST=[
                "/home/liuji/datasets/matting/sod/train01/"
    ]
# Paths of val datsets
SOD_VAL_PATHS_LIST = [
                "/home/liuji/datasets/matting/sod/val01/"
    ]
OTHER_DICT = {'animal':[ANIMAL_TRAIN_PATHS_LIST, ANIMAL_VAL_PATHS_LIST],
              'car':[CAR_TRAIN_PATHS_LIST, CAR_VAL_PATHS_LIST],
              'clothes':[CLOTHES_TRAIN_PATHS_LIST, CLOTHES_VAL_PATHS_LIST],
              'food':[FOOD_TRAIN_PATHS_LIST, FOOD_VAL_PATHS_LIST],
              'furniture':[FURNITURE_TRAIN_PATHS_LIST, FURNITURE_VAL_PATHS_LIST],
              'sod':[SOD_TRAIN_PATHS_LIST, SOD_VAL_PATHS_LIST],
              }    
              
######### Paths of pretrained model
PRETRAINED_R34_MP = ''
PRETRAINED_SWIN_STEM_POOLING5 = ''
PRETRAINED_VITAE_NORC_MAXPOOLING_BIAS_BASIC_STAGE4_14 = ''

########## Parameters for training
CROP_SIZE = 544
RESIZE_SIZE = 512
######### Test config
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
MIN_SIZE_H = 512
MIN_SIZE_W = 512
SHORTER_PATH_LIMITATION = 1080
