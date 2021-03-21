"""
    for many to one
"""
import os
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
from .base_sr_dataset import BaseVSRDataset
from .registry import DATASETS
from .utils import get_key_for_video_imgs
from edit.utils import scandir, is_list_of, mkdir_or_exist, is_tuple_of

IMG_EXTENSIONS = ('.png', )

@DATASETS.register_module()
class SRManyToOneDataset(BaseVSRDataset):
    """Many to One dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a list containing paired data [images, label].

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        mode (str): "train", "test" or "eval"
    """

    def __init__(self,
                 lq_folder,
                 pipeline,
                 gt_folder = "",
                 num_input_frames = 7,
                 scale = 4,
                 mode = "train",
                 eval_part = None):
        super(SRManyToOneDataset, self).__init__(pipeline, scale, mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.eval_part = eval_part
        if eval_part is not None:
            assert is_tuple_of(eval_part, str)

        self.data_infos = self.load_annotations()
        self.logger.info("SRManyToOneDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def load_annotations(self):
        # get keys
        keys = list(scandir(self.lq_folder, suffix=IMG_EXTENSIONS, recursive=True))
        keys = [ v for v in keys if len(v.split('/')) == 2]
        keys = sorted(keys, key=get_key_for_video_imgs)  # 000/00000.png
        
        # do split for train and eval
        if self.eval_part is not None:
            if self.mode == "train":
                keys = [k for k in keys if k.split('/')[0] not in self.eval_part]
            elif self.mode == "eval":
                keys = [k for k in keys if k.split('/')[0] in self.eval_part]
            else:
                pass

        self.frame_num = dict()
        for key in keys:
            self.frame_num[key.split("/")[0]] = 0
        for key in keys:
            self.frame_num[key.split("/")[0]] += 1

        data_infos = []
        for key in keys:
            # do some checks, to make sure the key for LR and HR is same. 
            if self.mode in ("train", "eval"):
                gt_path = os.path.join(self.gt_folder, key)
                assert os.path.exists(gt_path), "please make sure the key {} for LR and HR is same".format(key)

            if self.mode == "train":
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        LRkey=key,
                        HRkey=key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        gt_path = self.gt_folder,
                        LRkey = key,
                        HRkey = key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        LRkey = key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            else:
                raise NotImplementedError("")
        return data_infos
