#!/usr/bin/env python
# encoding: utf-8

# 加载需要的包
import argparse
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.model_5_2_x4_add import Net_rgb_text, Net_rgb_image, net_total
from loss.L1_loss import L1_loss
from loss.L1_loss_zero import L1_loss_zero
import math
import numpy as np
from data.dataset import DatasetFromFolder
from torch.utils.tensorboard import SummaryWriter

import os
import cv2
import sys
from tqdm import tqdm
import copy
from sewar.full_ref import psnr
from sewar.full_ref import ssim

# 设置代码加载目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

# 设置gpu使用哪一个
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--earlyLrStep", type=int, default=100, help="training batch size")
parser.add_argument("--lrReduce", type=int, default=0.8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=2000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained",
                    default="",
                    type=str)
parser.add_argument("--train_file", default=['./dataset/train/total_crop',
                                             './dataset/train/segment/seg_img_hr',
                                             './dataset/train/segment/seg_text_hr',
                                             './dataset/train/segment/seg_img_x4',
                                             './dataset/train/segment/seg_text_x4',
                                             './dataset/train/segment/seg_img_x2',
                                             './dataset/train/segment/seg_text_x2',
                                             './dataset/train/total_crop_x4'],
                    type=list)

parser.add_argument("--test_file", default=['./dataset/test/total_crop',
                                            './dataset/test/segment/seg_img_hr',
                                            './dataset/test/segment/seg_text_hr',
                                            './dataset/test/segment/seg_img_x4',
                                            './dataset/test/segment/seg_text_x4',
                                            './dataset/test/segment/seg_img_x2',
                                            './dataset/test/segment/seg_text_x2',
                                            './dataset/test/total_crop_x4'],
                    type=list)


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    log_dir = './log/save/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 1000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.enabled = True
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromFolder(opt.train_file[0], opt.train_file[1], opt.train_file[2], opt.train_file[3],
                                  opt.train_file[4], opt.train_file[5], opt.train_file[6], opt.train_file[7])
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    test_set = DatasetFromFolder(opt.test_file[0], opt.test_file[1], opt.test_file[2], opt.test_file[3],
                                 opt.test_file[4], opt.test_file[5], opt.test_file[6], opt.test_file[7])
    test_data_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1,
                                  shuffle=False)

    print("===> Building model")
    model_text = Net_rgb_text()
    model_image = Net_rgb_image()
    model = net_total(model_image, model_text)
    criterion = L1_loss()
    criterion_zero = L1_loss_zero()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_zero = criterion_zero.cuda()
    else:
        model = model.cpu()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}' ".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    min_psnr = 0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, criterion_zero, epoch, tb_writer)
        if epoch % 1 == 0:
            acc = eval(model, test_data_loader, epoch, tb_writer, cuda=True)  # add
            if acc > min_psnr:
                best_model = copy.deepcopy(model)
                min_psnr = acc
                save_best_checkpoint(best_model, epoch)
            else:
                save_latest_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch < opt.earlyLrStep:
        lr = opt.lr
    else:
        step = opt.step
        lr = opt.lr * (opt.lrReduce ** (epoch // step - 1))
    return lr


def train(training_data_loader, optimizer, model, criterion, criterion_zero, epoch, tb_writer):
    avg_loss = 0
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in tqdm(enumerate(training_data_loader, 1), desc='train dataset',
                                 total=len(training_data_loader)):

        input_img, input_text, x2_label, HR_label, hr_img, hr_text, hr_img_x2, hr_text_x2 = Variable(
            batch[0]), \
                                                                                            Variable(
                                                                                                batch[1]), \
                                                                                            Variable(
                                                                                                batch[2],
                                                                                                requires_grad=False), \
                                                                                            Variable(
                                                                                                batch[3],
                                                                                                requires_grad=False), \
                                                                                            Variable(
                                                                                                batch[4],
                                                                                                requires_grad=False), \
                                                                                            Variable(
                                                                                                batch[5],
                                                                                                requires_grad=False), \
                                                                                            Variable(
                                                                                                batch[6],
                                                                                                requires_grad=False), \
                                                                                            Variable(
                                                                                                batch[7],
                                                                                                requires_grad=False)

        if opt.cuda:
            input_img = input_img.cuda()
            input_text = input_text.cuda()
            x2_label = x2_label.cuda()
            HR_label = HR_label.cuda()
            hr_img = hr_img.cuda()
            hr_img_x2 = hr_img_x2.cuda()
            hr_text = hr_text.cuda()
            hr_text_x2 = hr_text_x2.cuda()

        HR_2x_text, HR_2x_img, HR_4x_text, HR_4x_img, x2, x4, edge = model(input_img, input_text)

        loss_img_x2 = criterion_zero(HR_2x_img, hr_img_x2)
        loss_text_x2 = criterion_zero(HR_2x_text, hr_text_x2)
        loss_x2_split = loss_img_x2 + loss_text_x2
        loss_x2_fusion = criterion(x2, x2_label)
        loss_x2 = 0.1 * loss_x2_split + loss_x2_fusion

        loss_img_x4 = criterion_zero(HR_4x_img, hr_img)
        loss_text_x4 = criterion_zero(HR_4x_text, hr_text)
        loss_x4_split = loss_img_x4 + loss_text_x4
        loss_x4_fusion = criterion(x4, HR_label)
        loss_x4 = 0.1 * loss_x4_split + loss_x4_fusion

        loss = loss_x2 + loss_x4

        optimizer.zero_grad()
        loss_x2.backward(retain_graph=True)
        loss_x4.backward()

        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): loss_x2: {:.10f}, loss_x4: {:.10f},loss: {:.10f}".format(epoch,
                                                                                                   iteration,
                                                                                                   len(
                                                                                                       training_data_loader),
                                                                                                   loss_x2.item(),
                                                                                                   loss_x4.item(),
                                                                                                   loss.item()))
        tb_writer.add_scalar('loss_x2', loss_x2, epoch)
        tb_writer.add_scalar('loss_x4', loss_x4, epoch)
        tb_writer.add_scalar('loss', loss, epoch)
        avg_loss += loss.item()
        torch.cuda.empty_cache()
    print('average loss ....................', avg_loss / len(training_data_loader))


def eval(model, test_data_loader, epoch, tb_writer, cuda=True):
    avg_psnr_predicted = []
    avg_ssim_predicted = []
    model.eval()
    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(test_data_loader, 1), desc='eval dataset', total=len(test_data_loader)):

            input_img, input_text, x2_label, HR_label, hr_img, hr_text, hr_img_x2, hr_text_x2 = Variable(
                batch[0]), \
                                                                                                Variable(
                                                                                                    batch[1]), \
                                                                                                Variable(
                                                                                                    batch[2],
                                                                                                    requires_grad=False), \
                                                                                                Variable(
                                                                                                    batch[3],
                                                                                                    requires_grad=False), \
                                                                                                Variable(
                                                                                                    batch[4],
                                                                                                    requires_grad=False), \
                                                                                                Variable(
                                                                                                    batch[5],
                                                                                                    requires_grad=False), \
                                                                                                Variable(
                                                                                                    batch[6],
                                                                                                    requires_grad=False), \
                                                                                                Variable(
                                                                                                    batch[7],
                                                                                                    requires_grad=False)

            if opt.cuda:
                input_img = input_img.cuda()
                input_text = input_text.cuda()

            HR_2x_text, HR_2x_img, HR_4x_text, HR_4x_img, x2, x4, edge = model(input_img, input_text)

            x4 = x4.mul(255).cpu().detach().numpy().squeeze(0)
            x4 = x4.squeeze(0)
            x4 = x4.astype(np.float32)
            HR_4x = x4

            HR_label = HR_label.mul(255).byte()
            HR_label = HR_label.cpu().numpy().squeeze(0)
            HR_label = HR_label.squeeze(0)
            label_x8 = HR_label

            if HR_4x.shape[1] != label_x8.shape[1] or HR_4x.shape[0] != label_x8.shape[0]:
                if HR_4x.shape[0] > label_x8.shape[0] or HR_4x.shape[1] > label_x8.shape[0]:
                    HR_4x = HR_4x[:label_x8.shape[0], :]
                    HR_4x = HR_4x[:, :label_x8.shape[1]]

                    HR_4x[HR_4x < 0] = 0.
                    HR_4x[HR_4x > 255.] = 255.

                    psnr_predicted = psnr(label_x8, HR_4x)
                    ssim_predicted = ssim(label_x8, HR_4x)[0]

                    avg_psnr_predicted.append(psnr_predicted)
                    avg_ssim_predicted.append(ssim_predicted)
            else:
                HR_4x[HR_4x < 0] = 0.
                HR_4x[HR_4x > 255.] = 255.

                psnr_predicted = psnr(label_x8, HR_4x)
                ssim_predicted = ssim(label_x8, HR_4x)[0]

                avg_psnr_predicted.append(psnr_predicted)
                avg_ssim_predicted.append(ssim_predicted)

    print('avg_psnr_predicted ....................', sum(avg_psnr_predicted) / len(test_data_loader))
    print('avg_ssim_predicted  ....................', sum(avg_ssim_predicted) / len(test_data_loader))
    tb_writer.add_scalar('avg_psnr_predicted', sum(avg_psnr_predicted) / len(test_data_loader), epoch)
    tb_writer.add_scalar('avg_ssim_predicted', sum(avg_ssim_predicted) / len(test_data_loader), epoch)

    total_psnr = sum(avg_psnr_predicted) / len(test_data_loader)

    return total_psnr


def save_latest_checkpoint(model, epoch):
    model_folder = "./dataset/trained_models/save/"
    model_out_path = model_folder + "latest_model_" + str(epoch) + ".pth"
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_best_checkpoint(model, epoch):
    model_folder = "./dataset/trained_models/save/"
    model_out_path = model_folder + "best_model_" + str(epoch) + ".pth"
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
