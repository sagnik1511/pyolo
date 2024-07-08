import torch
import torch.nn as nn


class Conv(nn.Module):

    def __init__(
        self, in_channels, out_chanels, kernel_size, stride, padding=1, eps=1e-5
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_chanels, eps=eps)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))
