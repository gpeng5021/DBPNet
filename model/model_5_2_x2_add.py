#!/usr/bin/env python
# encoding: utf-8


import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling
    https://blog.csdn.net/jiongnima/article/details/78578876"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output


class Net_rgb_text(nn.Module):
    def __init__(self):
        super(Net_rgb_text, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        return HR_2x


class Net_rgb_image(nn.Module):
    def __init__(self):
        super(Net_rgb_image, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        return HR_2x


class Edge_Enhancement(nn.Module):
    def __init__(self):
        super(Edge_Enhancement, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.stage_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage_1_conv_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.stage_1_conv_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool_1 = nn.AvgPool2d(2, stride=2)

        self.stage_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.stage_2_conv_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.stage_2_conv_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.pool_2 = nn.AvgPool2d(2, stride=2)

        self.final_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.stage_1(x))
        stage_1_conv_1 = self.relu(self.stage_1_conv_1(out))
        stage_1_conv_2 = self.relu(self.stage_1_conv_2(stage_1_conv_1))

        out_2 = self.pool_1(out)
        stage_2 = self.relu(self.stage_2(out_2))
        stage_2_conv_1 = self.relu(self.stage_2_conv_1(stage_2))
        stage_2_conv_2 = self.relu(self.stage_2_conv_2(stage_2_conv_1))
        deconv_2 = self.relu(self.deconv_2(stage_2_conv_2))

        feture_out = stage_1_conv_2 + deconv_2
        I_edge = feture_out

        return I_edge


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, edge, sr):
        out = torch.cat((edge, sr), 1)
        out = self.conv(out)

        return out


class net_total(nn.Module):
    def __init__(self, model_image, model_text):
        super(net_total, self).__init__()

        self.image = model_image
        self.text = model_text
        self.Edge_Enhancement = Edge_Enhancement()
        self.final_fusion = Fusion()

    def forward(self, x_img, x_text):
        HR_2x_text = self.text(x_text)
        HR_2x_img = self.image(x_img)

        HR_2x_text_edge = self.Edge_Enhancement(HR_2x_text)
        HR_2x_text__out = 0.1 * HR_2x_text_edge + HR_2x_text
        x2 = HR_2x_img + HR_2x_text__out

        return HR_2x_img, HR_2x_text__out, x2


if __name__ == '__main__':
    from torchsummary import summary

    model_text = Net_rgb_text()
    model_image = Net_rgb_image()
    model = net_total(model_image, model_text)

    img = torch.rand(1, 1, 132, 132)
    text = torch.rand(1, 1, 132, 132)
    summary(model, [(1, 10, 10), (1, 10, 10)])
