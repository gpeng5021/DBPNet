#!/usr/bin/env python
# encoding: utf-8

import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from PIL import Image

from torchvision.transforms import Compose, ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif'])


def image_to_tensor():
    return Compose([
        ToTensor(),
    ])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')

    y, _, _ = img.split()
    return y


def load_edge_img(filepath):
    img = Image.open(filepath)

    y = img
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_HR, image_dir_img, image_dir_text, image_dir_img_x2, image_dir_text_x2,
                 image_dir_LR_img, image_dir_LR_text, image_dir_x2):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_HR = [join(image_dir_HR, x) for x in listdir(image_dir_HR) if is_image_file(x)]
        self.image_filenames_LR_img = [join(image_dir_LR_img, x) for x in listdir(image_dir_LR_img) if is_image_file(x)]
        self.image_filenames_LR_text = [join(image_dir_LR_text, x) for x in listdir(image_dir_LR_text) if
                                        is_image_file(x)]
        self.image_filenames_x2 = [join(image_dir_x2, x) for x in listdir(image_dir_x2) if
                                   is_image_file(x)]
        self.image_filenames_img = [join(image_dir_img, x) for x in listdir(image_dir_img) if
                                    is_image_file(x)]
        self.image_filenames_text = [join(image_dir_text, x) for x in listdir(image_dir_text) if
                                     is_image_file(x)]
        self.image_filenames_img_x2 = [join(image_dir_img_x2, x) for x in listdir(image_dir_img_x2) if
                                       is_image_file(x)]
        self.image_filenames_text_x2 = [join(image_dir_text_x2, x) for x in listdir(image_dir_text_x2) if
                                        is_image_file(x)]

        self.image_to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        input_img = load_img(self.image_filenames_LR_img[index])
        input_img = self.image_to_tensor(input_img)

        input_text = load_img(self.image_filenames_LR_text[index])
        input_text = self.image_to_tensor(input_text)

        x8 = load_img(self.image_filenames_HR[index])
        x8 = self.image_to_tensor(x8)

        x2 = load_img(self.image_filenames_x2[index])
        x2 = self.image_to_tensor(x2)

        hr_img = load_img(self.image_filenames_img[index])
        hr_img = self.image_to_tensor(hr_img)

        hr_text = load_img(self.image_filenames_text[index])
        hr_text = self.image_to_tensor(hr_text)

        hr_img_x2 = load_img(self.image_filenames_img_x2[index])
        hr_img_x2 = self.image_to_tensor(hr_img_x2)

        hr_text_x2 = load_img(self.image_filenames_text_x2[index])
        hr_text_x2 = self.image_to_tensor(hr_text_x2)

        return input_img, input_text, x2, x8, hr_img, hr_text, hr_img_x2, hr_text_x2

    def __len__(self):
        return len(self.image_filenames_LR_img)


class test_DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_HR, image_dir_LR_img, image_dir_LR_text, image_dir_x2):
        super(test_DatasetFromFolder, self).__init__()
        self.image_filenames_HR = [join(image_dir_HR, x) for x in listdir(image_dir_HR) if is_image_file(x)]
        self.image_filenames_LR_img = [join(image_dir_LR_img, x) for x in listdir(image_dir_LR_img) if is_image_file(x)]
        self.image_filenames_LR_text = [join(image_dir_LR_text, x) for x in listdir(image_dir_LR_text) if
                                        is_image_file(x)]
        self.image_filenames_x2 = [join(image_dir_x2, x) for x in listdir(image_dir_x2) if
                                   is_image_file(x)]

        self.image_to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        input_img = load_img(self.image_filenames_LR_img[index])
        input_img = self.image_to_tensor(input_img)

        input_text = load_img(self.image_filenames_LR_text[index])
        input_text = self.image_to_tensor(input_text)

        x8 = load_img(self.image_filenames_HR[index])
        x8 = self.image_to_tensor(x8)

        x2 = load_img(self.image_filenames_x2[index])
        x2 = self.image_to_tensor(x2)

        return input_img, input_text, x2, x8

    def __len__(self):
        return len(self.image_filenames_LR_img)
