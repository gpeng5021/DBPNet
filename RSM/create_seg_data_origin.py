#!/usr/bin/env python
# encoding: utf-8

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
import os

from craft import CRAFT

from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=True, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='../dataset/total_crop/', type=str,
                    help='folder path to input images')

args = parser.parse_args()

""" For test images in a folder """


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def save_mask(image_path, image_list, split_name, k, polys, rename, textpath, imgpath, up_scale):
    result_folder_text = os.path.join(args.test_folder.split('total_crop')[0] + 'segment', textpath)
    result_folder_image = os.path.join(args.test_folder.split('total_crop')[0] + 'segment', imgpath)
    if not os.path.isdir(result_folder_text):
        os.mkdir(result_folder_text)
    if not os.path.isdir(result_folder_image):
        os.mkdir(result_folder_image)

    print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
    image_lr_path = os.path.join(image_path.split(split_name)[0] + rename, image_path.split(split_name)[1])

    image_lr = cv2.imread(image_lr_path)

    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    image = image_lr

    seg = []
    mask_white = []

    if len(polys) > 0:
        for i in range(len(polys)):
            pts = polys[i]
            pts = pts / up_scale
            pts = np.int32(pts)
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.polylines(mask, [pts], 1, 255)
            cv2.fillPoly(mask, [pts], 255)
            dst = cv2.bitwise_and(image, image, mask=mask)
            seg.append(dst)

            mask_white.append(mask)

        seg_text = seg[0]
        for j in range(len(seg)):
            if j + 1 < len(seg):
                seg_text = cv2.add(seg_text, seg[j + 1])

        seg_img = mask_white[0]
        for j in range(len(mask_white)):
            if j + 1 < len(mask_white):
                seg_img = cv2.add(seg_img, mask_white[j + 1])
        seg_img_mask = np.ones(image.shape[:2], np.uint8)
        seg_img_mask = cv2.bitwise_and(seg_img_mask, seg_img_mask, mask=seg_img)
        seg_img_mask_3C = cv2.cvtColor(seg_img_mask, cv2.COLOR_GRAY2BGR)

        for k in range(image.shape[2]):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if seg_img_mask_3C[i][j][k] == 0:
                        seg_img_mask_3C[i][j][k] = image[i][j][k]
                    else:
                        seg_img_mask_3C[i][j][k] = seg_img_mask_3C[i][j][k]

        seg_img = seg_img_mask_3C

        text_file = os.path.join(result_folder_text, filename + '.jpeg')
        image_file = os.path.join(result_folder_image, filename + '.jpeg')
        cv2.imwrite(image_file, seg_img)
        cv2.imwrite(text_file, seg_text)
    else:
        mask = np.zeros(image.shape[:2], np.uint8)
        text_file = os.path.join(result_folder_text, filename + '.jpeg')
        image_file = os.path.join(result_folder_image, filename + '.jpeg')
        cv2.imwrite(image_file, image)
        cv2.imwrite(text_file, mask)


if __name__ == '__main__':
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint {}'.format(args.trained_model))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # split_name = 'total_crop\\'  # windows
    split_name = 'total_crop/'  # ubuntu
    image_list, _, _ = file_utils.get_files(args.test_folder)
    # load data
    for k, image_path in enumerate(image_list):
        image = imgproc.loadImage(image_path)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly)
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        t = time.time()
        save_mask(image_path, image_list, split_name, k, polys, 'total_crop', 'seg_text_hr', 'seg_img_hr', 1.0)
        save_mask(image_path, image_list, split_name, k, polys, 'total_crop_x2', 'seg_text_x2', 'seg_img_x2', 4.0)
        save_mask(image_path, image_list, split_name, k, polys, 'total_crop_x4', 'seg_text_x4', 'seg_img_x4', 2.0)
        print('num: {}'.format(k))
        print("elapsed time : {}s".format(time.time() - t))
