#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import cv2
import os
from PIL import Image
import torchvision
from sewar.full_ref import psnr
from sewar.full_ref import ssim
from utils.log import Log

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")

parser.add_argument("--cuda", action="store_true", default=False, help="use cuda?")
parser.add_argument("--model",
                    default="./models/x2/best_model.pth",
                    type=str)
parser.add_argument("--dataset", default="./dataset/test/total_crop/", type=str)
opt = parser.parse_args()

model = torch.load(opt.model, map_location=torch.device('cpu'))["model"]

image_list = glob.glob(opt.dataset + "/*.*")

avg_psnr_predicted = []
avg_ssim_predicted = []
avg_elapsed_time = 0
i = 0

image_to_tensor = torchvision.transforms.ToTensor()

name_text = 'seg_text_x4'
name_img = 'seg_img_x4'
split_name = 'total_crop'
data_trained = 'save/'
save_path = './dataset/results/'
log_path = os.path.join(save_path, data_trained)
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_name = os.path.join(log_path + 'res.log')
log = Log(__name__, log_name)
logger = log.Logger

for image_name in image_list:
    i += 1
    logger.info("Processing %d, %s", i, image_name)
    img_pil = Image.open(image_name)
    width, height = img_pil.size

    img_hr_y, img_hr_cb, img_hr_cr = img_pil.convert("YCbCr").split()
    img_hr_cb = np.array(img_hr_cb)
    img_hr_cr = np.array(img_hr_cr)
    img_hr_y = np.array(img_hr_y)

    save_img_path = os.path.join(save_path + data_trained + 'res')
    im_path = os.path.join(save_img_path + image_name.split(split_name)[1])

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    text_lr = Image.open(
        os.path.join(image_name.split(split_name)[0]) + 'segment/' + name_text + image_name.split(split_name)[1])  # 修改
    img_lr = Image.open(
        os.path.join(image_name.split(split_name)[0]) + 'segment/' + name_img + image_name.split(split_name)[1])

    text_lr_y, _, _ = text_lr.convert("YCbCr").split()
    img_lr_y, _, _ = img_lr.convert("YCbCr").split()

    img_lr_y = Variable(image_to_tensor(img_lr_y).unsqueeze(0))
    text_lr_y = Variable(image_to_tensor(text_lr_y).unsqueeze(0))

    if opt.cuda:
        model = model.cuda()
        text_lr_y = text_lr_y.cuda()
        img_lr_y = img_lr_y.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_2x_img, HR_2x_text, x2 = model(img_lr_y, text_lr_y)

    HR_4x = x2
    HR_4x = HR_4x.mul(255).cpu().detach().numpy().squeeze(0)
    HR_4x = HR_4x.squeeze(0)
    HR_4x = HR_4x.astype(np.float32)

    HR = HR_4x  # 修改

    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    if HR.shape[1] != width or HR.shape[0] != height:
        if HR.shape[0] > height or HR.shape[1] > width:
            HR = HR[:height, :]
            HR = HR[:, :width]
            if HR.shape[0] == height and HR.shape[1] == width:

                HR[HR < 0] = 0.
                HR[HR > 255.] = 255.

                psnr_predicted = psnr(img_hr_y, HR)
                ssim_predicted = ssim(img_hr_y, HR)[0]

                logger.info('psnr_predicted %f ', psnr_predicted)
                logger.info('ssim_predicted %f ', ssim_predicted)

                total_img = np.zeros((HR.shape[0], HR.shape[1], 3), np.uint8)
                total_img[:, :, 0] = HR
                total_img[:, :, 1] = np.asarray(img_hr_cb)
                total_img[:, :, 2] = np.asarray(img_hr_cr)

                total_img_pil = Image.fromarray(total_img, "YCbCr").convert("RGB")
                total_img_pil.save(im_path)
                logger.info('write SR pic ok .....')

                avg_psnr_predicted.append(psnr_predicted)
                avg_ssim_predicted.append(ssim_predicted)
            else:
                print("Processing ", i, image_name)
                print(HR.shape, height, width)
    else:
        HR[HR < 0] = 0.
        HR[HR > 255.] = 255.

        psnr_predicted = psnr(img_hr_y, HR)
        ssim_predicted = ssim(img_hr_y, HR)[0]

        logger.info('psnr_predicted %f ', psnr_predicted)
        logger.info('ssim_predicted %f ', ssim_predicted)

        total_img = np.zeros((HR.shape[0], HR.shape[1], 3), np.uint8)
        total_img[:, :, 0] = HR
        total_img[:, :, 1] = np.asarray(img_hr_cb)
        total_img[:, :, 2] = np.asarray(img_hr_cr)

        total_img_pil = Image.fromarray(total_img, "YCbCr").convert("RGB")
        total_img_pil.save(im_path)
        logger.info('write SR pic ok .....')

        avg_psnr_predicted.append(psnr_predicted)
        avg_ssim_predicted.append(ssim_predicted)

logger.info("====================== Performance summary ======================")
logger.info(opt.dataset)
logger.info(f"PSNR: {sum(avg_psnr_predicted) / len(image_list):.5f}")
logger.info(f"SSIM: {sum(avg_ssim_predicted) / len(image_list):.5f}")
logger.info("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))
logger.info("============================== End ==============================")
