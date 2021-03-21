import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
import math

class SEL(M.Module):
    def __init__(self, hidden):
        super(SEL, self).__init__()
        self.conv = M.Conv2d(hidden, hidden, 3, 1, padding=1)
        self.relu = M.ReLU()

    def forward(self, x):
        return x * F.sigmoid( self.conv(self.relu(x)) )

def default_init_weights(module, scale=1, nonlinearity="relu"):
    """
        nonlinearity: leaky_relu
    """
    for m in module.modules():
        if isinstance(m, M.Conv2d):
            M.init.msra_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
            m.weight *= scale
            if m.bias is not None:
                M.init.zeros_(m.bias)
        else:       
            pass

class IMDModule(M.Module):
    def __init__(self, in_channels, c1= 20, c2=12, c3=4, distillation_rate=0.5):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = M.Conv2d(in_channels, c1, 3, stride=1, padding=1)
        self.c2 = M.Conv2d(c1 - self.distilled_channels, c2, 3, stride=1, padding=1)
        self.c3 = M.Conv2d(c2 - self.distilled_channels, c3, 3, stride=1, padding=1)
        self.act = M.LeakyReLU(negative_slope=0.1)
        self.c5 = M.Conv2d(2*self.distilled_channels + c3 + in_channels, in_channels, 1, stride=1, padding=0)
        self.sel =SEL(in_channels)
        # self.gct = GCT(2*self.distilled_channels + c3 + in_channels)
        self.init_weights()
        
    def forward(self, x):
        out_c1 = self.act(self.c1(x)) # 12 -> 27
        distilled_c1 = out_c1[:, :self.distilled_channels, :, :] # 3
        remaining_c1 = out_c1[:, self.distilled_channels: , :, :] # 24
        out_c2 = self.act(self.c2(remaining_c1)) #  24 -> 15
        distilled_c2 = out_c2[:, :self.distilled_channels, :, :] # 3
        remaining_c2 = out_c2[:, self.distilled_channels:, :, :] # 12
        out_c3 = self.act(self.c3(remaining_c2)) # 30 -> 12
        out = F.concat([distilled_c1, distilled_c2, out_c3, x], axis=1)
        out_fused = self.sel(self.c5(out))
        return out_fused

    def init_weights(self):
        for m in [self.c1, self.c2, self.c3]:
            default_init_weights(m, scale=0.2, nonlinearity='leaky_relu')

class Upsample(M.Module):
    def __init__(self, hidden):
        super(Upsample, self).__init__()
        self.shirking = M.Conv2d(4*hidden, hidden, kernel_size=1, stride=1, padding=0)
        self.sel = SEL(hidden)
        self.reconstruction = IMDModule(in_channels=hidden)
        self.conv_last = M.Conv2d(hidden, 3, kernel_size=3, stride=1, padding=1)
        self.conv_hr1 = M.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.conv_hr2 = M.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.lrelu = M.LeakyReLU(negative_slope=0.1)
        self.init_weights()

    def forward(self, inputs):
        out = self.sel(self.shirking(inputs))
        out = self.reconstruction(out)

        out = F.nn.interpolate(inp=out, scale_factor=2, mode='BILINEAR', align_corners=False)
        out = self.lrelu(self.conv_hr1(out))
        out = F.nn.interpolate(inp=out, scale_factor=2, mode='BILINEAR', align_corners=False)
        out = self.lrelu(self.conv_hr2(out))
        out = self.conv_last(out)
        return out

    def init_weights(self):
        for m in [self.conv_hr1, self.conv_hr2]:
            default_init_weights(m, nonlinearity='leaky_relu')


@BACKBONES.register_module()
class BasicVSR_v5(M.Module):
    def __init__(self, in_channels, 
                        out_channels, 
                        hidden_channels,
                        upscale_factor):
        super(BasicVSR_v5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upscale_factor = upscale_factor

        self.conv1 = M.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2)
        self.feature_extracter_rgb1 = IMDModule(in_channels=hidden_channels)
        self.feature_extracter_rgb2 = IMDModule(in_channels=hidden_channels)
        self.shirking1 = M.Conv2d(4*hidden_channels, 2*hidden_channels, kernel_size=1, stride=1, padding=0)
        self.feature_extracter_aggr1 = IMDModule(in_channels=2*hidden_channels)
        self.upsample = Upsample(hidden_channels) # need init

    def rgb(self, x):
        x = self.conv1(x)   # [B, 12, h, w]
        x1 = self.feature_extracter_rgb1(x)
        x2 = self.feature_extracter_rgb2(x1)
        out = F.concat([x1, x2], axis=1)
        return out

    def aggr(self, x):
        x = self.shirking1(x)
        x2 = self.feature_extracter_aggr1(x)
        return x2

    def init_weights(self, pretrained):
        pass
 
    def forward(x):
        return x