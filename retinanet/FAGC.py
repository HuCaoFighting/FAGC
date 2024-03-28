import torch
import torch.nn as nn

class FAGC(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FAGC, self).__init__()

        self.gray_conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dvs_conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.att_conv = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):

        residual = x1
        x1 = self.gray_conv(x1)
        x2 = self.dvs_conv(x2)

        x = x1 + x2
        x = self.att_conv(x)
        x_att = self.sigmoid(x)
        x = residual * x_att

        return x
