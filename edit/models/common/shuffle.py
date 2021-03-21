import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
import math
from . import default_init_weights

class ShuffleV2Block(M.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            M.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            M.ReLU(),
            # dw
            M.Conv2d(
                mid_channels, mid_channels, ksize, stride, pad,
                groups=mid_channels, bias=False,
            ),
            # pw-linear
            M.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            M.ReLU(),
        ]
        self.branch_main = M.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                M.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                M.BatchNorm2d(inp),
                # pw-linear
                M.Conv2d(inp, inp, 1, 1, 0, bias=False),
                M.BatchNorm2d(inp),
                M.ReLU(),
            ]
            self.branch_proj = M.Sequential(*branch_proj)
        else:
            self.branch_proj = None
        self.init_weights()

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return F.concat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return F.concat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        else:
            raise ValueError("use stride 1 or 2, current stride {}".format(self.stride))

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        # assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = F.transpose(x, (1, 0, 2))
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def init_weights(self):
        default_init_weights(self, scale=0.2)