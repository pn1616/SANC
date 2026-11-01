"""Experiment E: UNet++ (nested dense skip pathways)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv00 = ConvBlock(in_channels, 64)
        self.conv10 = ConvBlock(64, 128)
        self.conv20 = ConvBlock(128, 256)
        self.conv30 = ConvBlock(256, 512)
        self.conv40 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv01 = ConvBlock(64 + 128, 64)
        self.conv11 = ConvBlock(128 + 256, 128)
        self.conv21 = ConvBlock(256 + 512, 256)
        self.conv31 = ConvBlock(512 + 1024, 512)
        self.conv02 = ConvBlock(64 * 2 + 128, 64)
        self.conv12 = ConvBlock(128 * 2 + 256, 128)
        self.conv22 = ConvBlock(256 * 2 + 512, 256)
        self.conv03 = ConvBlock(64 * 3 + 128, 64)
        self.conv13 = ConvBlock(128 * 3 + 256, 128)
        self.conv04 = ConvBlock(64 * 4 + 128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def upsample(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        up, cat = self.upsample, torch.cat
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))
        x01 = self.conv01(cat([x00, up(x10)], 1))
        x11 = self.conv11(cat([x10, up(x20)], 1))
        x21 = self.conv21(cat([x20, up(x30)], 1))
        x31 = self.conv31(cat([x30, up(x40)], 1))
        x02 = self.conv02(cat([x00, x01, up(x11)], 1))
        x12 = self.conv12(cat([x10, x11, up(x21)], 1))
        x22 = self.conv22(cat([x20, x21, up(x31)], 1))
        x03 = self.conv03(cat([x00, x01, x02, up(x12)], 1))
        x13 = self.conv13(cat([x10, x11, x12, up(x22)], 1))
        x04 = self.conv04(cat([x00, x01, x02, x03, up(x13)], 1))
        return self.final(x04)
