###################################################################################################################
#  LMFFNet: A Well-Balanced Lightweight Network for Fast and Accurate Semantic Segmentation
#  Authors: M Shi, J Shen, Q Yi, J Weng, Z Huang, A Luo, Y Zhou
#  Published in£ºIEEE Transactions on Neural Networks and Learning Systems
#  Date: 2022/06/14
#
##################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LMFFNet"]


def Split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class conv3x3_resume(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv3x3 = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.conv1x1_resume = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1_resume(output)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Init_Block(nn.Module):
    def __init__(self):
        super(Init_Block, self).__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

    def forward(self, x):
        o = self.init_conv(x)
        return o


class SEM_B(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv_left = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                               padding=(1, 1), groups=nIn // 4, bn_acti=True)

        self.dconv_right = Conv(nIn // 4, nIn // 4, (dkSize, dkSize), 1,
                                padding=(1 * d, 1 * d), groups=nIn // 4, dilation=(d, d), bn_acti=True)


        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3_resume = conv3x3_resume(nIn , nIn , (dkSize, dkSize), 1,
                                padding=(1 , 1 ),  bn_acti=True)

    def forward(self, input):

        output = self.conv3x3(input)

        x1, x2 = Split(output)

        letf = self.dconv_left(x1)

        right = self.dconv_right(x2)

        output = torch.cat((letf, right), 1)
        output = self.conv3x3_resume(output)

        return self.bn_relu_1(output + input)


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class SENet_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SENet_Block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class PMCA(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(PMCA, self).__init__()

        self.partition_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv2x2 = Conv(ch_in, ch_in, 2, 1, padding=(0, 0), groups=ch_in, bn_acti=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.SE_Block = SENet_Block(ch_in=ch_in, reduction=reduction)

    def forward(self, x):
        o1 = self.partition_pool(x)

        o1 = self.conv2x2(o1)

        o2 = self.global_pool(x)

        o_sum = o1 + o2
        w = self.SE_Block(o_sum)
        o = w * x

        return o


class FFM_A(nn.Module):
    def __init__(self, ch_in):
        super(FFM_A, self).__init__()
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x
        o = self.bn_prelu(torch.cat([x1, x2], 1))
        o = self.conv1x1(o)
        return o


class FFM_B(nn.Module):
    def __init__(self, ch_in, ch_pmca):
        super(FFM_B, self).__init__()
        self.PMCA = PMCA(ch_in=ch_pmca, reduction=8)
        self.bn_prelu = BNPReLU(ch_in)
        self.conv1x1 = Conv(ch_in, ch_in, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2, x3 = x
        x2 = self.PMCA(x2)
        o = self.bn_prelu(torch.cat([x1, x2, x3], 1))
        o = self.conv1x1(o)
        return o


class SEM_B_Block(nn.Module):
    def __init__(self, num_channels, num_block, dilation, flag):
        super(SEM_B_Block, self).__init__()
        self.SEM_B_Block = nn.Sequential()
        for i in range(0, num_block):
            self.SEM_B_Block.add_module("SEM_Block_" + str(flag) + str(i), SEM_B(num_channels, d=dilation[i]))

    def forward(self, x):
        o = self.SEM_B_Block(x)
        return o


class MAD(nn.Module):
    def __init__(self, c1=16, c2=32, classes=19):
        super(MAD, self).__init__()
        self.c1, self.c2 = c1, c2
        self.LMFFNet_Block_2 = nn.Sequential()

        self.mid_layer_1x1 = Conv(128 + 3, c1, 1, 1, padding=0, bn_acti=False)

        self.deep_layer_1x1 = Conv(256 + 3, c2, 1, 1, padding=0, bn_acti=False)

        self.DwConv1 = Conv(self.c1 + self.c2, self.c1 + self.c2, (3, 3), 1, padding=(1, 1),
                            groups=self.c1 + self.c2, bn_acti=True)

        self.PwConv1 = Conv(self.c1 + self.c2, classes, 1, 1, padding=0, bn_acti=False)

        self.DwConv2 = Conv(256 + 3, 256 + 3, (3, 3), 1, padding=(1, 1), groups=256 + 3, bn_acti=True)
        self.PwConv2 = Conv(256 + 3, classes, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x

        x2_size = x2.size()[2:]

        x1_ = self.mid_layer_1x1(x1)
        x2_ = self.deep_layer_1x1(x2)

        x2_ = F.interpolate(x2_, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        x1_x2_cat = torch.cat([x1_, x2_], 1)
        x1_x2_cat = self.DwConv1(x1_x2_cat)
        x1_x2_cat = self.PwConv1(x1_x2_cat)
        x1_x2_cat_att = torch.sigmoid(x1_x2_cat)

        o = self.DwConv2(x2)
        o = self.PwConv2(o)
        o = F.interpolate(o, [x2_size[0] * 2, x2_size[1] * 2], mode='bilinear', align_corners=False)

        o = o * x1_x2_cat_att

        o = F.interpolate(o, [x2_size[0] * 8, x2_size[1] * 8], mode='bilinear', align_corners=False)

        return o


class LMFFNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=8):
        super().__init__()
        self.classes = classes
        self.block_1 = block_1
        self.block_2 = block_2
        self.Init_Block = Init_Block()

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.FFM_A = FFM_A(32 + 3)

        self.downsample_1 = DownSamplingBlock(32 + 3, 64)

        self.SEM_B_Block1 = SEM_B_Block(num_channels=64, num_block=self.block_1, dilation=[2, 2, 2], flag=1)

        self.FFM_B1 = FFM_B(ch_in=128 + 3, ch_pmca=64)

        self.downsample_2 = DownSamplingBlock(128 + 3, 128)

        self.SEM_B_Block2 = SEM_B_Block(num_channels=128, num_block=self.block_2, dilation=[4, 4, 8, 8, 16, 16, 32, 32], flag=2)

        self.FFM_B2 = FFM_B(ch_in=256 + 3, ch_pmca=128)

        self.MAD = MAD(classes=self.classes)

    def forward(self, input):
        # Init Block
        out_init_block = self.Init_Block(input)
        down_1 = self.down_1(input)
        input_ffm_a = out_init_block, down_1

        # FFM-A
        out_ffm_a = self.FFM_A(input_ffm_a)

        # SEM-B Block1
        out_downsample_1 = self.downsample_1(out_ffm_a)
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)

        # FFM-B1
        down_2 = self.down_2(input)
        input_sem1_pmca1 = out_sem_block1, out_downsample_1, down_2
        out_ffm_b1 = self.FFM_B1(input_sem1_pmca1)

        # SEM-B Block2
        out_downsample_2 = self.downsample_2(out_ffm_b1)
        out_se_block2 = self.SEM_B_Block2(out_downsample_2)

        # FFM-B2
        down_3 = self.down_3(input)
        input_sem2_pmca2 = out_se_block2, out_downsample_2, down_3
        out_ffm_b2 = self.FFM_B2(input_sem2_pmca2)

        # MAD
        input_ffmb1_ffmb2 = out_ffm_b1, out_ffm_b2
        out_mad = self.MAD(input_ffmb1_ffmb2)

        return out_mad

