import random
import numpy as np
import cv2
from megengine.data.transform import ColorJitter as mge_color_jitter
# from megengine.data.transform import ContrastTransform as mge_contrast
from edit.utils import imwrite
from ..registry import PIPELINES

def brightness(image, value):
    if value == 0:
        return image
    dtype = image.dtype
    image = image.astype(np.float32)
    alpha = value
    image = image * alpha
    return image.clip(0, 255).astype(dtype)

@PIPELINES.register_module()
class Add_brightness(object):
    def __init__(self, keys, value_sar = 1, value_optical = 1):
        self.keys = keys
        self.value_sar = value_sar
        self.value_optical = value_optical

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                raise NotImplementedError("not support list key")
            else:
                if key in ['sar', 'SAR']:
                    results[key] = brightness(results[key], value = self.value_sar)
                elif key in ['optical', 'OPTICAL', 'opt', "OPT"]:
                    # imwrite(results[key], "./workdirs/{}_origin.png".format(self.nums))
                    results[key] = brightness(results[key], value = self.value_optical)
                    # imwrite(results[key], "./workdirs/{}_hou.png".format(self.nums))
                    # self.nums += 1
                else:
                    raise NotImplementedError("not support key")
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


def contrast(image, value):
    if value <= 0:
        return image
    dtype = image.dtype
    image = image.astype(np.float32)
    alpha = value
    image = image * alpha + image.mean() * (1 - alpha)
    return image.clip(0, 255).astype(dtype)


@PIPELINES.register_module()
class Add_contrast(object):
    def __init__(self, keys, value_sar = 1, value_optical = 1):
        self.keys = keys
        self.value_sar = value_sar
        self.value_optical = value_optical

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                raise NotImplementedError("not support list key")
            else:
                if key in ['sar', 'SAR']:
                    results[key] = contrast(results[key], value = self.value_sar)
                elif key in ['optical', 'OPTICAL', 'opt', "OPT"]:
                    # imwrite(results[key], "./workdirs/{}_origin.png".format(self.nums))
                    results[key] = contrast(results[key], value = self.value_optical)
                    # imwrite(results[key], "./workdirs/{}_hou.png".format(self.nums))
                    # self.nums += 1
                else:
                    raise NotImplementedError("not support key")
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


def bgr2gray(img, keepdim=True):
    """Convert a BGR image to grayscale image.
    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.
    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


@PIPELINES.register_module()
class Bgr2Gray(object):
    def __init__(self, keys, keepdim=True):
        self.keys = keys
        self.keep_dim = keepdim

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    bgr2gray(v, self.keep_dim) for v in results[key]
                ]
            else:
                results[key] = bgr2gray(results[key], self.keep_dim)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self, keys, brightness=0.3, contrast=0.3, saturation=0.3, hue=0):
        self.keys = keys
        self.colorjitter = mge_color_jitter(brightness, contrast, saturation, hue)
        # self.nums = 0

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    self.colorjitter.apply(v) for v in results[key]
                ]
            else:
                # imwrite(results[key], "./workdirs/{}_origin.png".format(self.nums))
                results[key] = self.colorjitter.apply(results[key])
                # imwrite(results[key], "./workdirs/{}_hou.png".format(self.nums))
                # self.nums += 1
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string