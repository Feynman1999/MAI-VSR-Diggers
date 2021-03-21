import tensorflow as tf
from tensorflow import keras
import numpy as np

class SEL(keras.Model):
    def __init__(self, hidden):
        super(SEL, self).__init__()
        self.conv = keras.layers.Conv2D(hidden, 3, 1, padding="same")
        self.relu = keras.layers.ReLU()

    def call(self, x):
        return x * keras.activations.sigmoid( self.conv(self.relu(x)) )

class Upsample(keras.Model): # use bilinear
    def __init__(self, hidden):
        super(Upsample, self).__init__()
        self.conv_hr1 = keras.layers.Conv2D(hidden, 3, padding="same", activation=keras.layers.ReLU(negative_slope=0.1))
        self.conv_hr2 = keras.layers.Conv2D(hidden, 3, padding="same", activation=keras.layers.ReLU(negative_slope=0.1))
        self.conv_last = keras.layers.Conv2D(3, 3, 1, padding="same")
        self.reconstruction = IMDModule(in_channels=hidden)
        self.sel = SEL(hidden)
        self.shirking = keras.layers.Conv2D(hidden, 1, 1, padding="same")

    def call(self, inputs):
        out, h, w = inputs
        out = self.sel(self.shirking(out))
        out = self.reconstruction(out)

        out = tf.image.resize(out, [h*2, w*2])
        out = self.conv_hr1(out)
        out = tf.image.resize(out, [h*4, w*4])
        out = self.conv_hr2(out)
        out = self.conv_last(out)
        return out

class IMDModule(keras.Model):
    def __init__(self, in_channels, c1=20, c2=12, c3=4, distillation_rate=0.5):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate) # 4
        self.remaining_channels = int(in_channels - self.distilled_channels) # 4
        self.c1 = keras.layers.Conv2D(c1, 3, padding="same", activation=keras.layers.ReLU(negative_slope=0.1)) # 8->20       16->20
        self.c2 = keras.layers.Conv2D(c2, 3, padding="same", activation=keras.layers.ReLU(negative_slope=0.1)) # 16->12      12->12
        self.c3 = keras.layers.Conv2D(c3, 3, padding="same", activation=keras.layers.ReLU(negative_slope=0.1)) # 8->4        4->4
        self.c5 = keras.layers.Conv2D(in_channels, 1, padding="same") # 4+4+4+8 = 20                                       8+8+4+16 = 36
        self.sel =SEL(in_channels)
        self.concat_c = keras.layers.Concatenate(axis=-1)

    def call(self, x):
        out_c1 = self.c1(x)
        distilled_c1 = out_c1[:, :, :, 0:self.distilled_channels]
        remaining_c1 = out_c1[:, :, :, self.distilled_channels:]  # 24
        out_c2 = self.c2(remaining_c1)
        distilled_c2 = out_c2[:, :, :, 0:self.distilled_channels] # 3
        remaining_c2 = out_c2[:, :, :, self.distilled_channels:] # 12
        out_c3 = self.c3(remaining_c2)
        out = self.concat_c([distilled_c1, distilled_c2, out_c3, x])
        out_fused = self.sel(self.c5(out))
        return out_fused

def get_bilinear(image, h, w):
    return tf.image.resize(image, [h*4, w*4])

class BidirectionalRestorer_V6(keras.Model):
    def __init__(self, hidden_channels = 8):
        super(BidirectionalRestorer_V6, self).__init__()

        self.hidden_channels = hidden_channels
        self.conv1 = keras.layers.Conv2D(hidden_channels, 5, 1, padding="same")
        self.feature_extracter_aggr1 = IMDModule(in_channels=2*hidden_channels)
        self.feature_extracter_rgb1 = IMDModule(in_channels=hidden_channels)
        self.feature_extracter_rgb2 = IMDModule(in_channels=hidden_channels)
        self.concat_c = keras.layers.Concatenate(axis = -1)
        self.concat_b = keras.layers.Concatenate(axis = 0)
        self.shirking1 = keras.layers.Conv2D(2*hidden_channels, 1, 1, padding="same") # 32 -> 16
        self.upsample = Upsample(hidden_channels)
        

    def rgb(self, x):
        x = self.conv1(x)   # [B, 12, h, w]
        x1 = self.feature_extracter_rgb1(x)
        x2 = self.feature_extracter_rgb2(x1)
        return self.concat_c([x1, x2])

    def aggr(self, x):
        x = self.shirking1(x)
        x1 = self.feature_extracter_aggr1(x)
        return x1

    def call(self, inputs):
        # inputs: [1, 180, 320, 30]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        biup = get_bilinear(inputs, h, w)
        B = 1
        T = 10
        inputs = tf.reshape(inputs, (h, w, T, 3))
        inputs = tf.transpose(inputs, (2, 0, 1, 3)) # [10, 180, 320, 3]
        inputs = self.rgb(inputs) # [10, 180, 320, hidden]
        
        res = []
        res2 = []

        now_frame_forward = inputs[0:1, :, :, :]
        now_frame_backward = inputs[T-1:, :, :, :]
        now_frame = self.concat_b([now_frame_forward, now_frame_backward]) # [2, h, w ,hidden]
        hidden = now_frame
        res.append(hidden)
        for i in range(1,T):
            now_frame_forward = inputs[i:i+1, :, :, :]
            now_frame_backward = inputs[(T-i-1):(T-i), :, :, :]
            now_frame = self.concat_b([now_frame_forward, now_frame_backward]) # [2, h, w ,hidden]
            hidden = self.aggr(self.concat_c([hidden, now_frame]))
            res.append(hidden)
        # upsample  list of [2, h, w, hidden]
        gap = 1
        for i in range(0, T, gap): #  [0, 2, 4, 6, 8]
            t = []
            for j in range(gap):
                t.append(self.concat_c([res[i + j][0:1, :, :, :], res[T-i-j-1][1:2, :, :, :]]))
            t_0_1_res = self.upsample([self.concat_b(t), h, w]) # [2, h, w, 2*hidden]
            for j in range(gap):
                res2.append(t_0_1_res[j:j+1, :, :, :])
        res = self.concat_c(res2) # [1,H,W,3*T]
        return res + biup
        # return tf.clip_by_value(res + biup, 0, 1) # need clip to 0~1
