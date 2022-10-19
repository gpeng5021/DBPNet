#!/usr/bin/env python
# encoding: utf-8

import cv2
import glob
import os

path = './dataset/total_crop_no_border'
out_path = './dataset/total_crop/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
image_list = glob.glob(path + "/*.*")

i = 0
resolution = 128
for image_name in image_list:
    i += 1
    print("Processing ", i, image_name)
    out_name = out_path + image_name.split('total_crop_no_border\\')[1]
    img = cv2.imread(image_name)
    width, height = img.shape[1], img.shape[0]
    if width > 128 or height > 128:
        print("Processing ", i, image_name)
    else:
        a = cv2.copyMakeBorder(img, int((resolution - height) / 2), int((resolution - height) / 2),
                               int((resolution - width) / 2),
                               int((resolution - width) / 2), cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if a.shape[0] != resolution or a.shape[1] != resolution:
            out = cv2.resize(a, (resolution, resolution), cv2.INTER_CUBIC)
            cv2.imwrite(out_name, out)
        else:
            cv2.imwrite(out_name, a)
