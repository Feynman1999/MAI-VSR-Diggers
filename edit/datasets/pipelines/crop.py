import random
import math
import os
import numpy as np
import cv2
from ..registry import PIPELINES
from edit.utils import imresize, imwrite
from skimage.segmentation import slic, mark_boundaries


@PIPELINES.register_module()
class MinimumBoundingBox_ByOpticalFlow(object):
    def __init__(self, blocksizes = [9, 8], n_segments = (50, 70), compactness = (10,20)):
        self.blocksizes = blocksizes
        self.n_segments = n_segments  # for skimage slic
        self.compactness = compactness  # for skimage slic
        self.flow_dir = "/data/home/songtt/chenyuxiang/datasets/REDS/train/train_sharp_bicubic/X4_RAFT_sintel"
        self.name_padding_len = 8
        self.max_pixel_num = 48**2
        self.threthld = 8*9
        self.scale = 4

    def check_valid(self, x):
        if x[0]<0 or x[0] >= 180:
            return False
        if x[1]<0 or x[1] >= 320:
            return False
        return True     

    def viz(self, mask, idx, tl, br, clipname, img):
        c = np.zeros((180,320))
        for h,w in mask:
            c[h,w]=1
        c = (c*255).astype(np.uint8)
        cv2.rectangle(c, (tl[1], tl[0]), (br[1], br[0]), (255,0,0), 1)
        cv2.rectangle(img, (tl[1], tl[0]), (br[1], br[0]), (255,0,0), 1)
        imwrite(c, "./{}/{}_mask.png".format(clipname, idx))
        imwrite(img, "./{}/{}_img.png".format(clipname, idx))

    def __call__(self, results):
        """
            crop lq and gt frames
            steps:
            1. for first frame of lq, do slic algorithm(compactness = 10, and n_segments uniformly select), and randomly select one area
            2. use optical flow information, crop along the motion trajectory (for lq frames) (different frames
            may be have different crop size, up to object spatial size, but all are integral multiple of the block size, for transformer easy training)
            the length may be reduce (because object spatial size is be too small)  (e.g. 30 -> 15)
            3. paired crop gt frames
            4. add meta information of location (for transformer position encoding)
        """
        clipname = os.path.dirname(results['LRkey'])
        lq_paths = results['lq_path']

        n_segments = random.randint(self.n_segments[0], self.n_segments[1])
        compactness = random.randint(self.compactness[0], self.compactness[1])
        segments_lq_first_frame = slic(results['lq'][0], n_segments=n_segments, compactness=compactness, start_label=0)
        # out1=mark_boundaries(results['lq'][0], segments_lq_first_frame)
        # imwrite((out1*255).astype(np.uint8), "./{}.png".format(clipname))
        
        # randomly select one area
        max_class_id = np.max(segments_lq_first_frame)
        select_class_id = random.randint(0, max_class_id)
        lq_masks = []
        mask_lq_first_frame = np.argwhere(select_class_id == segments_lq_first_frame)  # e.g. (672, 2) int64
        while mask_lq_first_frame.shape[0] > self.max_pixel_num:
            select_class_id = random.randint(0, max_class_id)
            mask_lq_first_frame = np.argwhere(select_class_id == segments_lq_first_frame)
        lq_masks.append(mask_lq_first_frame)

        first_frame_idx = os.path.basename(lq_paths[0])
        first_frame_idx = int(os.path.splitext(first_frame_idx)[0])
        for idx in range(first_frame_idx, first_frame_idx + len(lq_paths) - 1):
            # according    idx -> idx+1      flow   solve   idx+1  mask
            flowpath = os.path.join(self.flow_dir, clipname, "{}_{}.npy".format(str(idx).zfill(self.name_padding_len), str(idx+1).zfill(self.name_padding_len)))
            flow = np.load(flowpath)
            L = []
            for h,w in lq_masks[-1]:
                res = [int(flow[h,w,1]+0.5) + h, int(flow[h,w,0]+0.5) + w]
                if self.check_valid(res):
                    L.append(res)
            if len(L) < self.threthld:
                break
            new_mask = np.array(L)
            lq_masks.append(new_mask)

        # crop for lq and gt
        lq_crops = []
        gt_crops = []
        for idx in range(0, len(lq_masks)):
            tl = np.min(lq_masks[idx], axis=0) # top-left
            br = np.max(lq_masks[idx], axis=0) # bottom-right
            # make tl and br    are integral multiple of the block size
            tl = (np.floor(tl / self.blocksizes) * self.blocksizes ).astype(np.int64)
            br = (np.ceil((br+1) / self.blocksizes) * self.blocksizes ).astype(np.int64) - 1
            # viz
            # self.viz(lq_masks[idx], idx, tl, br, clipname, results['lq'][idx])
            length_h = br[0] - tl[0] + 1
            length_w = br[1] - tl[1] + 1
            lq_crops.append(results['lq'][idx][tl[0]:tl[0] + length_h, tl[1]:tl[1] + length_w, ...])
            gt_crops.append(results['gt'][idx][self.scale * tl[0]:self.scale * (tl[0] + length_h), self.scale*tl[1]: self.scale*(tl[1] + length_w), ...])
            # print(lq_crops[-1].shape, gt_crops[-1].shape)
        results['lq'] = lq_crops
        results['gt'] = gt_crops
        return results


@PIPELINES.register_module()
class Random_Crop_Opt_Sar(object):
    def __init__(self, keys, size, have_seed = False, Contrast=False):
        self.keys = keys
        self.size = size # 500, 320
        self.have_seed = have_seed
        self.Contrast = Contrast
        if self.Contrast:
            assert have_seed == False

    def get_optical_h_w(self, sar_h, sar_w):
        up = 800 - self.size[0]
        optical_h = random.randint(max(sar_h - (self.size[0]-self.size[1]), 0), min(sar_h, up))
        optical_w = random.randint(max(sar_w - (self.size[0]-self.size[1]), 0), min(sar_w, up))
        return optical_h, optical_w

    def __call__(self, results):
        if self.have_seed:  # 用于测试时
            random.seed(np.sum(results['sar']))

        gap = 512 - self.size[1]
        sar_h = random.randint(0, gap) # 随机两个数 去裁剪sar
        sar_w = random.randint(0, gap)
        # 获得sar图像
        results['sar'] = results['sar'][sar_h:sar_h+self.size[1], sar_w:sar_w+self.size[1], :]  # h,w,1
        # 所以我们可以得到裁剪出的sar图在800中的左上角
        sar_h = results['bbox'][0] + sar_h
        sar_w = results['bbox'][1] + sar_w
        
        if self.Contrast: # 随机三个用于训练
            sar = results['sar']
            optical = results['opt']
            results['sar'] = []
            results['opt'] = []
            results['bbox'] = []
            for _ in range(3):
                results['sar'].append(sar.copy())
                opt = optical.copy()
                optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
                results['opt'].append(opt[optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :])
                results['bbox'].append(np.array([sar_h - optical_h, 
                                                 sar_w - optical_w, 
                                                 sar_h - optical_h + self.size[1] - 1, 
                                                 sar_w - optical_w + self.size[1] - 1]).astype(np.float32))
        else:
            optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
            results['opt'] = results['opt'][optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :]  # h,w,1

            # 更改bbox
            results['bbox'][0] = sar_h - optical_h
            results['bbox'][1] = sar_w - optical_w
            results['bbox'][2] = results['bbox'][0] + self.size[1] - 1
            results['bbox'][3] = results['bbox'][1] + self.size[1] - 1
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class PairedRandomCrop(object):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        gt_patch_size ([int, int]): cropped gt patch size.
    """

    def __init__(self, gt_patch_size, fix0=False, crop_flow=False):
        self.crop_flow = crop_flow
        if isinstance(gt_patch_size, int):
            self.gt_patch_h = gt_patch_size
            self.gt_patch_w = gt_patch_size
        else:
            self.gt_patch_h, self.gt_patch_w = gt_patch_size

        self.fix0 = fix0

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = results['scale']
        assert self.gt_patch_h % scale == 0 and self.gt_patch_w % scale == 0
        lq_patch_h = self.gt_patch_h // scale
        lq_patch_w = self.gt_patch_w // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise RuntimeError("HR's size is not {}X times to LR's size".format(scale))
            # do resize, resize gt to lq * scale
            # results['gt'] = [
            #     imresize(v, (w_lq * scale, h_lq * scale))
            #     for v in results['gt']
            # ]
            
        if h_lq < lq_patch_h or w_lq < lq_patch_w:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_h}, {lq_patch_w}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        if self.fix0:
            top = 0
            left = 0
        else:
            top = random.randint(0, h_lq - lq_patch_h)
            left = random.randint(0, w_lq - lq_patch_w)


        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_h, left:left + lq_patch_w, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_h,
              left_gt:left_gt + self.gt_patch_w, ...] for v in results['gt']
        ]

        # crop flow
        if self.crop_flow:
            # results['flow']    list of [h,w,2] 
            results['flow'] = [
                v[top:top + lq_patch_h, left:left + lq_patch_w, ...]
                for v in results['flow']
            ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_h}, {self.gt_patch_w})'
        return repr_str
