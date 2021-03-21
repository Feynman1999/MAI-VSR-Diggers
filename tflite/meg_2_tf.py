import megengine as mge
from net_v6 import BidirectionalRestorer_V6
from tensorflow import keras
import os

def load_from_meg(netG, path):
    assert os.path.exists(path)    
    state_dict = mge.load(path)
    mge_params = [ state_dict[key] for key in state_dict.keys()]
    tf_params = []
    for i in range(len(mge_params)):
        if i%2==0: # need kernel
            res = mge_params[i+1].transpose((2,3,1,0))
        else:  # need bias
            val = mge_params[i-1].shape[1]
            res = mge_params[i-1].reshape((val, ))
        tf_params.append(res)
    netG.set_weights(tf_params)
    return netG


if __name__ == "__main__":

    # for weight, v in zip(netG.get_weights(),netG.trainable_weights):
    path = "./workdirs/v6/generator_module.mge"
    state_dict = mge.load(path)
