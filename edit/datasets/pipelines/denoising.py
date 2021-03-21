import random
import cv2
import numpy as np
from edit.utils import imwrite
from ..registry import PIPELINES

@PIPELINES.register_module()
class NLmeanDenoising(object):
    def __init__(self, keys, h=1, kernel=7, search = 21):
        self.keys = keys
        self.h = h
        self.kernel = kernel
        self.search = search

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                raise NotImplementedError("")
            else:
                results[key] = cv2.fastNlMeansDenoising(results[key], None, h=self.h, templateWindowSize =self.kernel, searchWindowSize = self.search)
                if len(results[key].shape) == 2:
                    results[key] = np.expand_dims(results[key], axis=2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
