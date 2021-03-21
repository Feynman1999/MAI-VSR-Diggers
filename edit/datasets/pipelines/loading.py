import os
from ..registry import PIPELINES
from edit.utils import FileClient, imfrombytes
import numpy as np
import pickle

def get_bin_path(path):
    suffix = path.split(".")[-1]
    bin_path = path[:-len(suffix)] + "pkl"
    return bin_path

def read_bin(bin_path):
    with open(bin_path, 'rb') as _f:
        img = pickle.load(_f)
    return img

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 save_original_img=False,
                 make_bin=False,
                 **kwargs):
        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.make_bin = make_bin # 注意使用make_bin之前请先使用单gpu 单进程模式跑一个epoch，确保所有文件都已经创建bin
        self.file_client = None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepath = str(results[f'{self.key}_path'])
        img_bytes = self.file_client.get(filepath)
        img = imfrombytes(img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []

        for filepath in filepaths:
            if self.make_bin:
                bin_path = get_bin_path(filepath)
                if os.path.isfile(bin_path):
                    img = read_bin(bin_path)
                else:
                    raise NotImplementedError("please make sure all pkl file exist first")
                    # img_bytes = self.file_client.get(filepath)
                    # img = imfrombytes(img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC, BGR
                    # make_ndarray_bin(img, bin_path)
            else:
                img_bytes = self.file_client.get(filepath)
                img = imfrombytes(img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC, BGR
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            imgs.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs
        
        return results
