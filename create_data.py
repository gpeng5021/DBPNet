#!/usr/bin/env python
# encoding: utf-8
import glob
import math
import os

import torch.utils.data
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from tqdm import tqdm
import numpy as np


def calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    """Some operations of making data set. Reference from `https://github.com/xinntao/BasicSR`"""

    if (scale < 1) and antialiasing:
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    # print(img.shape)
    # exit()
    in_C, in_H, in_W = img.size()
    if in_C != 3:
        # print('aaaaaaaaaa')
        # print('rrrr', img.shape)
        new_img = np.zeros([3, img.shape[1], img.shape[2]], np.uint8)
        # img = img.numpy()
        # img = np.expand_dims(img, axis=0)
        # img = np.concatenate((img, img, img), axis=0)
        new_img[0, :, :] = img[0, :, :]
        new_img[1, :, :] = img[0, :, :]
        new_img[2, :, :] = img[0, :, :]
        img = torch.tensor(new_img)
        # print('ssss', img.shape)
    in_C, in_H, in_W = img.size()

    # in_H, in_W = img.size
    # _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    out_H, out_W = math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(in_H,
                                                                             out_H,
                                                                             scale,
                                                                             kernel_width,
                                                                             antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(in_W,
                                                                             out_W,
                                                                             scale,
                                                                             kernel_width,
                                                                             antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)

    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return torch.clamp(out_2, 0, 1)


pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()


def create_lr_data(input_dir, output_dir, upscale_factor):
    r""" The high resolution data set is preliminarily processed.
    """
    image_list = glob.glob(input_dir + "/*.*")
    for filename in tqdm(image_list, desc="Generating images from hr dir"):
        # print(filename)
        # exit()
        img = Image.open(filename)
        img = pil2tensor(img)

        # Save high resolution img
        # img = tensor2pil(img)
        # img.save(os.path.join(output_dir, os.path.basename(filename)), "bmp")
        # Simple down sampling.
        img = imresize(img, 1.0 / upscale_factor, True)

        # Save low resolution img
        img = tensor2pil(img)
        if os.path.exists(output_dir):
            img.save(os.path.join(output_dir, os.path.basename(filename)))
        else:
            os.mkdir(output_dir)
            img.save(os.path.join(output_dir, os.path.basename(filename)))


if __name__ == '__main__':
    input_dir = './dataset/total_crop'
    output_dir = './dataset/total_crop_x2'
    upscale_factor = 4

    create_lr_data(input_dir, output_dir, upscale_factor)
