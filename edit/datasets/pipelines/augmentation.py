import numpy as np
import os.path as osp
from ..registry import PIPELINES
from edit.utils import imflip_, bboxflip_, img_shelter, flowflip_


@PIPELINES.register_module()
class Corner_Shelter(object):
    def __init__(self, keys, shelter_ratio=0.1, black_ratio = 0.75):
        self.keys = keys
        self.shelter_ratio = shelter_ratio
        self.black_ratio = black_ratio

    def __call__(self, results):
        shelter = np.random.random() < self.shelter_ratio

        if shelter:
            for key in self.keys:
                if isinstance(results[key], list):
                    raise NotImplementedError("")
                else:
                    results[key] = img_shelter(results[key], self.black_ratio)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')
        return repr_str

@PIPELINES.register_module()
class RandomTransposeHW(object):
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if key in ('bbox', 'bboxes'):
                    if isinstance(results[key], list):
                        for idx, v in enumerate(results[key]):
                            tmp_bbox = v.copy()
                            tmp_bbox[0], tmp_bbox[1] = tmp_bbox[1], tmp_bbox[0]
                            tmp_bbox[2], tmp_bbox[3] = tmp_bbox[3], tmp_bbox[2]
                            results[key][idx] = tmp_bbox
                        # raise NotImplementedError("not imple for key is list for bbox tranposeHW")
                    else:
                        tmp_bbox = results[key].copy()
                        tmp_bbox[0], tmp_bbox[1] = tmp_bbox[1], tmp_bbox[0]
                        tmp_bbox[2], tmp_bbox[3] = tmp_bbox[3], tmp_bbox[2]
                        results[key] = tmp_bbox
                elif key in ('flow'):
                    assert isinstance(results[key], list)
                    tmp = []
                    for item in results[key]:
                        item1 = item.transpose(1,0,2)
                        tmp.append(-1 * np.stack([item1[:,:,1], item1[:, :, 0]], axis=2))
                    results[key] = tmp
                else:
                    if isinstance(results[key], list):
                        results[key] = [v.transpose(1, 0, 2) for v in results[key]]
                    else:
                        results[key] = results[key].transpose(1, 0, 2)

        results['transpose'] = transpose

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')
        return repr_str


@PIPELINES.register_module()
class Flip(object):
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal', Len=400):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.Len = Len

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        if key in ('bbox', 'bboxes'):
                            bboxflip_(v, self.direction, self.Len)
                        elif key in ('flow'):
                            flowflip_(v, self.direction)
                        else:
                            imflip_(v, self.direction)
                else:
                    if key in ('bbox', 'bboxes'):
                        bboxflip_(results[key], self.direction, self.Len)
                    elif key in ('flow'):
                        raise NotImplementedError("not implemented flow for just one frame")
                    else:
                        imflip_(results[key], self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str


@PIPELINES.register_module()
class GenerateFrameIndiceswithPadding(object):
    """Generate frame index with padding for Many to One or Many to Many dataset when test and eval.

    Required keys: lq_path, gt_path, key, num_input_frames, max_frame_num
    Added or modified keys: lq_path, gt_path

    Args:
         padding (str): padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'.

            Examples: current_idx = 0, num_input_frames = 5
            The generated frame indices under different padding mode:

                replicate: [0, 0, 0, 1, 2]
                reflection: [2, 1, 0, 1, 2]
                reflection_circle: [4, 3, 0, 1, 2]
                circle: [3, 4, 0, 1, 2]
    """

    def __init__(self, padding, many2many = False, index_start = 0, name_padding = True, dist_gap = 0):
        if padding not in ('replicate', 'reflection', 'reflection_circle', 'circle'):
            raise ValueError(f'Wrong padding mode {padding}.'
                             'Should be "replicate", "reflection", '
                             '"reflection_circle",  "circle"')
        self.padding = padding
        self.many2many = many2many
        self.index_start = index_start
        self.name_padding = name_padding
        self.dist_gap = dist_gap

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['LRkey'].split('/') # 000/0000001.png
        frame_name, ext_name = osp.splitext(frame_name)
        if self.name_padding:
            padding_length = len(frame_name)
        else:
            padding_length = 0
        current_idx = int(frame_name) - self.index_start  # start from 0, easy to cal
        max_frame_num = results['max_frame_num'] - 1  
        num_input_frames = results['num_input_frames']
        num_pad = num_input_frames // 2

        frame_list = []
        for i in range(current_idx - num_pad, current_idx + num_pad + 1):
            if i < 0:
                if self.padding == 'replicate':
                    pad_idx = 0
                elif self.padding == 'reflection':
                    pad_idx = -i
                elif self.padding == 'reflection_circle':
                    pad_idx = current_idx + num_pad - i
                else:
                    pad_idx = num_input_frames + i
            elif i > max_frame_num:
                if self.padding == 'replicate':
                    pad_idx = max_frame_num
                elif self.padding == 'reflection':
                    pad_idx = max_frame_num * 2 - i
                elif self.padding == 'reflection_circle':
                    pad_idx = (current_idx - num_pad) - (i - max_frame_num)
                else:
                    pad_idx = i - num_input_frames
            else:
                pad_idx = i
            frame_list.append(pad_idx)
        
        # add dist frames
        if self.dist_gap > 0:
            # select one frame every dist_gap frames
            ref_index = []
            for i in range(0, max_frame_num+1, self.dist_gap):
                if not (i in frame_list):
                    ref_index.append(i)
            frame_list += ref_index

        lq_path_root = results['lq_path']
        lq_paths = [
            osp.join(lq_path_root, clip_name,  str(idx + self.index_start).zfill(padding_length) + ext_name)
            for idx in frame_list
        ]
        results['lq_path'] = lq_paths
        
        # for eval 
        if 'HRkey' in results.keys():
            clip_name_HR, _ = results['HRkey'].split('/')
            gt_path_root = results['gt_path']
            if self.many2many:
                gt_paths = [
                    osp.join(gt_path_root, clip_name_HR, str(idx + self.index_start).zfill(padding_length) + ext_name)
                    for idx in frame_list
                ]
            else:
                gt_paths = [osp.join(gt_path_root, clip_name_HR, frame_name + ext_name)]
            results['gt_path'] = gt_paths
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(padding='{self.padding}')"
        return repr_str


@PIPELINES.register_module()
class STTN_REDS_GenerateFrameIndices(object):
    def __init__(self, interval_list, gap = 20):
        self.interval_list = interval_list
        self.gap = gap # 前后的间隔

    def __call__(self, results):
        clip_name, frame_name = results['LRkey'].split('/')  # key example: 000/00000000.png
        clip_name_HR, _ = results['HRkey'].split('/')  # key example: 000/00000000.png
        frame_name, ext_name = osp.splitext(frame_name)  # 00000000    .png
        padding_length = len(frame_name)
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2
        interval = np.random.choice(self.interval_list)
        start_frame_idx = center_frame_idx - num_half_frames * interval  # ensure not exceeding the borders
        end_frame_idx = center_frame_idx + num_half_frames * interval
        start = 0
        end = start + results['max_frame_num']
        while (start_frame_idx < start) or (end_frame_idx >= end):
            center_frame_idx = np.random.randint(start, end)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval

        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        if self.gap >0:
            # append to neighbor_list two frames (for reds)
            now_end = neighbor_list[-1]
            add_end_1 = min(now_end + self.gap, end-1)
            add_end_2 = min(now_end + self.gap*2, end-1)
            neighbor_list.append(add_end_1)
            neighbor_list.append(add_end_2)

            now_start = neighbor_list[0]
            add_start_1 = max(0, now_start - self.gap)
            # neighbor_list.insert(0, add_start_1)
            neighbor_list.append(add_start_1)

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_paths = [
            osp.join(lq_path_root, clip_name, str(v).zfill(padding_length) + ext_name)
            for v in neighbor_list
        ]
        gt_paths = [
            osp.join(gt_path_root, clip_name_HR, str(v).zfill(padding_length) + ext_name)
            for v in neighbor_list
        ]
        results['lq_path'] = lq_paths
        results['gt_path'] = gt_paths
        results['interval'] = interval
        return results

@PIPELINES.register_module()
class GenerateFrameIndices(object):
    """
        Generate frame index for many to many or many to one datasets. (for training)
        It also performs temporal augmention with random interval.

        Required keys: lq_path, gt_path, key, num_input_frames, max_frame_num
        Added or modified keys:  lq_path, gt_path, interval
    """

    def __init__(self, interval_list, many2many = False, index_start = 0, name_padding = True, load_flow = False): # default is REDS dataset
        self.interval_list = interval_list
        self.many2many = many2many
        self.index_start = index_start
        self.name_padding = name_padding
        self.load_flow = load_flow
        self.flow_dir = "/work_base/datasets/REDS/train/train_sharp_bicubic/X4_RAFT_sintel"

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['LRkey'].split('/')  # key example: 000/00000000.png
        clip_name_HR, _ = results['HRkey'].split('/')  # key example: 000/00000000.png
        frame_name, ext_name = osp.splitext(frame_name)  # 00000000    .png
        if self.name_padding:
            padding_length = len(frame_name)
        else:
            padding_length = 0
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        start = self.index_start
        end = start + results['max_frame_num']
        while (start_frame_idx < start) or (end_frame_idx >= end):
            center_frame_idx = np.random.randint(start, end)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval

        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_paths = [
            osp.join(lq_path_root, clip_name, str(v).zfill(padding_length) + ext_name)
            for v in neighbor_list
        ]
        if self.many2many:
            gt_paths = [
                osp.join(gt_path_root, clip_name_HR, str(v).zfill(padding_length) + ext_name)
                for v in neighbor_list
            ]
        else:
            frame_name = str(center_frame_idx).zfill(padding_length)
            gt_paths = [osp.join(gt_path_root, clip_name_HR, frame_name + ext_name)]
        results['lq_path'] = lq_paths
        results['gt_path'] = gt_paths
        results['interval'] = interval

        if self.load_flow:
            flows = []
            flow_paths = [
                osp.join(self.flow_dir, clip_name, str(v).zfill(padding_length)+"_"+ str(v+1).zfill(padding_length) + ".npy")
                for v in neighbor_list[:-1]
            ]
            # load npy
            for flowpath in flow_paths:
                flows.append(np.load(flowpath))
            results['flow'] = flows
            
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f', interval_list={self.interval_list}'
        return repr_str


@PIPELINES.register_module()
class TemporalReverse(object):
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The propability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key].reverse()

        results['reverse'] = reverse

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, reverse_ratio={self.reverse_ratio})'
        return repr_str