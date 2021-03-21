import megengine as mge
import megengine.module as M
import megengine.functional as F
import numpy as np
import math


class Flatten(M.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class ChannelGate(M.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()

        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        self.gate_c= M.Sequential(
            Flatten(),
            M.Linear(gate_channels[0], gate_channels[1]),
            M.BatchNorm1d(gate_channels[1]) ,
            M.ReLU(),
            M.Linear(gate_channels[-2], gate_channels[-1])
        )
        
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, (in_tensor.shape[2],in_tensor.shape[3]), stride=(in_tensor.shape[2],in_tensor.shape[3]))
        x = self.gate_c(avg_pool)
        x = F.expand_dims(x, axis=[2,3])  # b,48,1
        x = F.broadcast_to(x, in_tensor.shape)  # b,48,h,w
        return x

class SpatialGate(M.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = M.Sequential(
            M.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1),
            M.BatchNorm2d(gate_channel//reduction_ratio),
            M.ReLU(),
            M.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3, \
                     padding=dilation_val, dilation=dilation_val),
            M.BatchNorm2d(gate_channel // reduction_ratio),
            M.ReLU(),
            M.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3, \
                     padding=dilation_val, dilation=dilation_val),
            M.BatchNorm2d(gate_channel // reduction_ratio),
            M.ReLU(),
            M.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1))
    def forward(self, in_tensor):
        x = self.gate_s( in_tensor )
        x = F.broadcast_to(x,in_tensor.shape)
        return x

class BAM(M.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor


# if __name__ == '__main__':
#     a = mge.tensor(np.random.random((24,48,160, 160)).astype('float32'))
#     B =  BAM(gate_channel=48)
#     x = B.forward(a)
#     print(x.shape)