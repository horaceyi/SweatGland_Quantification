import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, out_ch),  # keep same as original
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, out_ch),  # keep same as original
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # keep same as original: conv1(x) is computed twice
        return self.conv2(self.conv1(x)) + self.conv1(x)


class DepthNet(nn.Module):
    def __init__(self, in_ch, out_ch, numchannel):
        super(DepthNet, self).__init__()

        self.num_channel = numchannel

        # Encoder
        self.conv1 = DoubleConv(in_ch, numchannel)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(numchannel, numchannel * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(numchannel * 2, numchannel * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(numchannel * 4, numchannel * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(numchannel * 8, numchannel * 16)
        self.pool5 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv6 = DoubleConv(numchannel * 16, numchannel * 16)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2)

        self.conv7 = DoubleConv(numchannel * 32, numchannel * 8)

        self.up7 = nn.ConvTranspose2d(
            numchannel * 8,
            numchannel * 4,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.conv8 = DoubleConv(numchannel * 16, numchannel * 4)

        self.up8 = nn.ConvTranspose2d(
            numchannel * 4,
            numchannel * 2,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.conv9 = DoubleConv(numchannel * 8, numchannel * 2)

        self.up9 = nn.ConvTranspose2d(
            numchannel * 2,
            numchannel,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.conv10 = DoubleConv(numchannel * 4, numchannel)

        self.conv11 = nn.Conv2d(2 * numchannel, out_ch, 1)
        self.conv12 = nn.Conv2d(out_ch, out_ch, 1)

        # Unused multi-level layers, kept for checkpoint compatibility
        self.mlconv7 = DoubleConv(8 * numchannel, out_ch)
        self.mlconv8 = DoubleConv(4 * numchannel, out_ch)
        self.mlconv9 = DoubleConv(2 * numchannel, out_ch)
        self.mlconv10 = DoubleConv(numchannel, out_ch)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        p5 = self.pool5(c5)

        c6 = self.conv6(p5)

        up_6 = self.up6(c6)
        merge6 = torch.cat([up_6, c5], dim=1)
        c7 = self.conv7(merge6)

        # Original code reuses self.up6 instead of self.up7
        up_7 = self.up6(c7)
        merge7 = torch.cat([up_7, c4], dim=1)
        c8 = self.conv8(merge7)

        # Original code reuses self.up6 instead of self.up8
        up_8 = self.up6(c8)
        merge8 = torch.cat([up_8, c3], dim=1)
        c9 = self.conv9(merge8)

        # Original code reuses self.up6 instead of self.up9
        up_9 = self.up6(c9)
        merge9 = torch.cat([up_9, c2], dim=1)
        c10 = self.conv10(merge9)

        up_10 = self.up6(c10)
        merge10 = torch.cat([up_10, c1], dim=1)
        c11 = self.conv11(merge10)

        output = nn.Softmax(dim=1)(c11)

        return output