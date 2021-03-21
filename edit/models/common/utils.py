import megengine
import megengine.module as M
import megengine.functional as F

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