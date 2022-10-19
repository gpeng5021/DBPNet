#!/usr/bin/env python
# encoding: utf-8


import cv2
from tqdm import tqdm
import random
import os
import numpy as np


def moveFile(fileDir, randint):
    random.seed(randint)
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = 0.25
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    img_list = []
    for name in sample:
        img = fileDir + name
        img_list.append(img)

    return img_list


def horizontal_flip(img_path):
    img = cv2.imread(img_path)
    h_flip = cv2.flip(img, 1)
    return h_flip


def vertical_flip(img_path):
    img = cv2.imread(img_path)
    v_flip = cv2.flip(img, 0)
    return v_flip


def hv_flip(img_path):
    img = cv2.imread(img_path)
    hv_flip = cv2.flip(img, -1)
    return hv_flip


def rotation(img_path):
    img = cv2.imread(img_path)
    rows, cols = img.shape[:2]
    degree = [90, 180, 270]
    slice = random.sample(degree, 3)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), slice[0], 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    return dst


def affine(img_path):
    img = cv2.imread(img_path)

    rows, cols = img.shape[:2]
    pts = []

    point1 = np.float32([[50, 50], [300, 50], [50, 200]])
    point2 = np.float32([[10, 100], [300, 50], [100, 250]])
    point3 = np.float32([[20, 35], [60, 70], [80, 21]])
    point4 = np.float32([[32, 78], [230, 40], [60, 250]])
    point5 = np.float32([[9, 66], [85, 32], [71, 210]])
    point6 = np.float32([[56, 82], [400, 50], [88, 99]])
    pts.append(point1)
    pts.append(point2)
    pts.append(point3)
    pts.append(point4)
    pts.append(point5)
    pts.append(point6)

    pts1 = random.sample(pts, 2)
    M = cv2.getAffineTransform(pts1[0], pts1[1])
    dst = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    return dst


def save_ro(randint=1111):
    in_hr_path = './dataset/train300_crop/'
    out_hr_path = './dataset/train300_crop_ro/'
    image_hr_list = moveFile(in_hr_path, randint)
    for filename in tqdm(image_hr_list, desc="Generating images from hr dir"):
        hr_filename = filename
        out_basename = os.path.basename(filename).split('.')[0] + '_ro' + '.png'

        out_hr_filename = os.path.join(out_hr_path, out_basename)
        img_hr = rotation(hr_filename)

        if os.path.exists(out_hr_path):
            cv2.imwrite(out_hr_filename, img_hr)
        else:
            os.mkdir(out_hr_path)
            cv2.imwrite(out_hr_filename, img_hr)


def save_hv(randint=111):
    in_hr_path = './dataset/train300_crop/'
    out_hr_path = './dataset/train300_crop_hv/'
    image_hr_list = moveFile(in_hr_path, randint)
    for filename in tqdm(image_hr_list, desc="Generating images from hr dir"):
        hr_filename = filename
        out_basename = os.path.basename(filename).split('.')[0] + '_hv' + '.png'

        out_hr_filename = os.path.join(out_hr_path, out_basename)
        img_hr = hv_flip(hr_filename)

        if os.path.exists(out_hr_path):
            cv2.imwrite(out_hr_filename, img_hr)
        else:
            os.mkdir(out_hr_path)
            cv2.imwrite(out_hr_filename, img_hr)


def save_ve(randint=11):
    in_hr_path = './dataset/train300_crop/'
    out_hr_path = './dataset/train300_crop_ve/'
    image_hr_list = moveFile(in_hr_path, randint)
    for filename in tqdm(image_hr_list, desc="Generating images from hr dir"):
        hr_filename = filename
        out_basename = os.path.basename(filename).split('.')[0] + '_ve' + '.png'

        out_hr_filename = os.path.join(out_hr_path, out_basename)
        img_hr = vertical_flip(hr_filename)

        if os.path.exists(out_hr_path):
            cv2.imwrite(out_hr_filename, img_hr)
        else:
            os.mkdir(out_hr_path)
            cv2.imwrite(out_hr_filename, img_hr)


def save_ho(randint=1):
    in_hr_path = './dataset/train300_crop/'
    out_hr_path = './dataset/train300_crop_ho/'
    image_hr_list = moveFile(in_hr_path, randint)
    for filename in tqdm(image_hr_list, desc="Generating images from hr dir"):
        hr_filename = filename
        out_basename = os.path.basename(filename).split('.')[0] + '_ho' + '.png'

        out_hr_filename = os.path.join(out_hr_path, out_basename)
        img_hr = horizontal_flip(hr_filename)

        if os.path.exists(out_hr_path):
            cv2.imwrite(out_hr_filename, img_hr)
        else:
            os.mkdir(out_hr_path)
            cv2.imwrite(out_hr_filename, img_hr)


if __name__ == '__main__':
    save_ro(randint=1111)
    save_hv(randint=111)
    save_ve(randint=11)
    save_ho(randint=1)
