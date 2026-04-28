import os
import random

img_dir_list = ["/data/liuji/datasets/nsfw_img/2025-04-16-sq/", 
                "/data/liuji/datasets/nsfw_img/2025-04-16-lhm(0415)/",
                "/data/liuji/datasets/nsfw_img/2025-04-16-cat/",
                "/data/liuji/datasets/nsfw_img/2025-03-31-classified/"
                ]
train_f = open("train.txt", 'w')
val_f = open("val.txt", 'w')
for img_dir in img_dir_list:
    for l in os.listdir(img_dir):
        tmp_dir = os.path.join(img_dir, l)
        for name in os.listdir(tmp_dir):
            if name.endswith('.jpg'):
                img_path = os.path.join(tmp_dir, name)
                item_info = "{} {} \n".format(img_path, l)
                if random.random() > 0.05:
                    train_f.write(item_info)
                else:
                    val_f.write(item_info)
train_f.close()
val_f.close()