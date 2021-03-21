import megengine.functional as F
import megengine.module as M
import megengine as mge
import numpy as np

class FilterResponseNormNd(M.Module):
    def __init__(self, ndim, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [4, ], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1, ) * (ndim - 2)
        self.eps = mge.tensor(np.ones(shape, dtype=np.float32)*eps)
        # if not learnable_eps:
        #     self.eps.requires_grad = False
        
        self.gamma = mge.Parameter(np.ones(shape, dtype=np.float32))
        self.beta = mge.Parameter(np.zeros(shape, dtype=np.float32))
        self.tau = mge.Parameter(np.zeros(shape, dtype=np.float32))
    
    def forward(self, x):
        B, C, _, _ = x.shape
        # avg_dims = tuple(range(2, len(x.shape)))  # [2 ,3 ]
        nu2 = F.expand_dims(F.pow(x, 2).reshape(B, C, -1).mean(axis=-1, keepdims=True), axis=-1)  # [B, C, 1, 1]
        x = x  / F.sqrt(nu2 + F.abs(self.eps))
        return F.maximum(self.gamma * x + self.beta, self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(3, num_features, eps=eps, learnable_eps=learnable_eps)

class FilterResponseNorm2d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(4, num_features, eps=eps, learnable_eps=learnable_eps)

class FilterResponseNorm3d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(5, num_features, eps=eps, learnable_eps=learnable_eps)