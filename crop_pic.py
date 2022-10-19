#!/usr/bin/env python
# encoding: utf-8

import os
from PIL import Image
import random
import numpy as np
import cv2
from tqdm import tqdm
import glob


def img_seg(hr_path, hr_out):
    if not os.path.exists(hr_out):
        os.makedirs(hr_out)
    files = os.listdir(hr_path)
    for file in files:
        a, b = os.path.splitext(file)
        hr_img = cv2.imread(os.path.join(hr_path + "\\" + file))
        hr_hight, hr_width = hr_img.shape[:2]

        id = 0
        while True:
            ratio = random.random()
            scale = 0.5 + ratio * (0.5 - 0.3)
            aa = random.random()
            bb = random.random()
            hr_new_h = int(hr_hight * scale * aa)
            hr_new_w = int(hr_width * scale * bb)

            if 0 < hr_new_w < hr_hight and 0 < hr_new_h < hr_width:
                y = np.random.randint(1, hr_hight - hr_new_h)
                x = np.random.randint(1, hr_width - hr_new_w)
                hr_new_img = hr_img[y:y + hr_new_h, x:x + hr_new_w, :]
                print(hr_out + a + "_" + str(id) + b)
                hr_out_path = hr_out + a + "_" + str(id) + b
                if 16 < hr_new_w < 128 and 16 < hr_new_h < 128:
                    if cv2.mean(hr_new_img)[0] < 240:
                        cv2.imwrite(hr_out_path, hr_new_img)
                id += 1

                if id == 230:
                    break


def crop_equal(image_path, out_path):
    image_list = glob.glob(image_path + "/*.*")
    for filename in tqdm(image_list, desc="Generating images from hr dir"):
        print(filename)
        img = Image.open(filename)
        size_img = img.size
        # print(size_img)
        x = 0
        y = 0
        w = int(size_img[0] / 2)
        h = int(size_img[1] / 2)
        id = 0
        for k in range(2):
            for v in range(2):
                region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
                basename = os.path.basename(filename).split('.')[0] + '_%d' % id + '.png'
                path = os.path.join(out_path, basename)
                # print(path)

                region.save(path)
                id += 1
        # exit()


if __name__ == '__main__':
    hr_path = "./dataset/train300/"
    hr_out = "./dataset/train300_crop/"

    img_seg(hr_path, hr_out)
