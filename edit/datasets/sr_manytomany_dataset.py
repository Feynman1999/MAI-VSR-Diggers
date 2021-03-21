"""
    for many to many
"""
import os
from .base_sr_dataset import BaseVSRDataset
from .registry import DATASETS
from .utils import get_key_for_video_imgs
from edit.utils import scandir, mkdir_or_exist, is_tuple_of, is_list_of

IMG_EXTENSIONS = ('.png', )

@DATASETS.register_module()
class SRManyToManyDataset(BaseVSRDataset):
    """
        Many to Many dataset for video super resolution.
    """
    def __init__(self,
                 lq_folder,
                 pipeline,
                 gt_folder = "",
                 num_input_frames = 11,
                 scale = 4,
                 mode = "train",
                 eval_part = None):
        super(SRManyToManyDataset, self).__init__(pipeline, scale, mode)
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
        self.logger.info("SRManyToManyDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def load_annotations(self):
        # get keys
        keys = list(scandir(self.lq_folder, suffix=IMG_EXTENSIONS, recursive=True))
        keys = [ v for v in keys if len(v.split('/')) == 2]
        keys = sorted(keys, key=get_key_for_video_imgs)  # 000/00000002.png for reds
        
        # do split for train and eval
        if self.eval_part is not None:
            if self.mode == "train":
                keys = [k for k in keys if k.split('/')[0] not in self.eval_part]
            else: # eval和test都会只选择部分
                keys = [k for k in keys if k.split('/')[0] in self.eval_part]

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
                        lq_path = self.lq_folder,
                        gt_path = self.gt_folder,
                        LRkey = key,
                        HRkey = key,
                        max_frame_num = self.frame_num[key.split("/")[0]],
                        num_input_frames = self.num_input_frames
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        gt_path = self.gt_folder,
                        LRkey = key,
                        HRkey = key,
                        max_frame_num = self.frame_num[key.split("/")[0]],                   
                        num_input_frames = self.num_input_frames
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        LRkey = key,
                        max_frame_num = self.frame_num[key.split("/")[0]],                   
                        num_input_frames = self.num_input_frames
                    )
                )
            else:
                raise NotImplementedError("")
        return data_infos
