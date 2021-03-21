import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from edit.models.common import ShuffleV2Block, CoordAtt
import math
from . import default_init_weights

# class GCT(M.Module):
#     def __init__(self, num_channels, epsilon=1e-5):
#         super(GCT, self).__init__()
#         self.alpha = megengine.Parameter(np.ones((1, num_channels, 1, 1), dtype=np.float32))
#         self.gamma = megengine.Parameter(np.zeros((1, num_channels, 1, 1), dtype=np.float32))
#         self.beta = megengine.Parameter(np.zeros((1, num_channels, 1, 1), dtype=np.float32))
#         self.epsilon = epsilon

#     def forward(self, x):
#         embedding = ((F.sum((x**2), axis=[2,3], keepdims=True) + self.epsilon)**(0.5)) * self.alpha
#         norm = self.gamma / ((F.mean((embedding**2), axis=1, keepdims=True) + self.epsilon)**(0.5))
#         gate = 1. + F.tanh(embedding * norm + self.beta)
#         return x * gate

# class SEL(M.Module):
#     def __init__(self, hidden):
#         super(SEL, self).__init__()
#         self.conv = M.Conv2d(hidden, hidden, 1, 1)
#         self.relu = M.ReLU()
#         self.init_weights()

#     def forward(self, x):
#         return x * F.sigmoid( self.conv(self.relu(x)) )

#     def init_weights(self):
#         default_init_weights(self.conv, scale=0.1)

class MobileNeXt(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
            默认使用coordinate attention在第一个dwise之后
            https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
        """
        super(MobileNeXt, self).__init__()
        self.dconv1 = M.ConvRelu2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), groups=in_channels)
        self.CA = CoordAtt(inp = out_channels, oup=out_channels)
        self.conv1 = M.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = M.ConvRelu2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.dconv2 = M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), groups=out_channels)
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2, self.dconv1, self.dconv2]:
            default_init_weights(m, scale=0.1)

    def forward(self, x):
        identity = x
        out = self.dconv2(self.conv2(self.conv1(self.CA(self.dconv1(x)))))
        return identity + out

class ResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = M.ConvRelu2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.conv2 = M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, scale=0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        return identity + out

class ResBlocks(M.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3, blocktype="resblock"):
        super(ResBlocks, self).__init__()
        assert blocktype in ("resblock", "shuffleblock", "MobileNeXt")
        if blocktype == "resblock":
            self.model = M.Sequential(
                self.make_resblock_layer(channel_num, resblock_num, kernel_size),
            )
        elif blocktype == "shuffleblock":
            self.model = M.Sequential(
                self.make_shuffleblock_layer(channel_num, resblock_num, kernel_size),
            )
        elif blocktype == "MobileNeXt":
            self.model = M.Sequential(
                self.make_MobileNeXt_layer(channel_num, resblock_num, kernel_size)
            )
        else:
            raise NotImplementedError("")

    def make_MobileNeXt_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(MobileNeXt(ch_out, ch_out, kernel_size))
        return M.Sequential(*layers)

    def make_resblock_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(ch_out, ch_out, kernel_size))
        return M.Sequential(*layers)

    def make_shuffleblock_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(ShuffleV2Block(inp = ch_out//2, oup=ch_out, mid_channels=ch_out//2, ksize=kernel_size, stride=1))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

