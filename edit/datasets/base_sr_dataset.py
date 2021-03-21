import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import copy
from collections import defaultdict
from .base_dataset import BaseDataset
from pathlib import Path
from edit.utils import scandir, is_list_of, mkdir_or_exist, is_tuple_of, imread, imwrite


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')


class BaseSRDataset(BaseDataset):
    """Base class for image super resolution Dataset.
    """

    def __init__(self, pipeline, scale, mode="train"):
        super(BaseSRDataset, self).__init__(pipeline, mode)
        self.scale = scale

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = sorted(list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True)))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)

    def evaluate(self, results):
        """Evaluate with different metrics.

        Args:
            results (list of dict): for every dict, record metric -> value for one frame

        Return:
            dict: Evaluation results dict.
        """
        assert is_list_of(results, dict), f'results must be a list of dict, but got {type(results)}'
        assert len(results) >= len(self), "results length should >= dataset length, due to multicard eval"
        self.logger.info("eval samples length: {}, dataset length: {}, only select front {} results".format(len(results), len(self), len(self)))
        results = results[:len(self)]

        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_results[metric].append(val)
        for metric, val_list in eval_results.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_results = {
            metric: sum(values) / len(self)
            for metric, values in eval_results.items()
        }

        return eval_results


class BaseVSRDataset(BaseDataset):
    """Base class for video super resolution Dataset.
    """

    def __init__(self, pipeline, scale, mode="train"):
        super(BaseVSRDataset, self).__init__(pipeline, mode)
        self.scale = scale

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.pipeline(results)

    def test_aggre(self, save_path, padding_len = 4, start_index = 1):
        clip_names = sorted(self.frame_num.keys())  # e.g. [`city`, `walk`]
        frame_nums = [ self.frame_num[clip] for clip in clip_names ]

        do_frames = 0
        now_clip_idx = 0
        total_deal = 0
        for _ in range(len(self)):
            do_frames += 1
            if do_frames == frame_nums[now_clip_idx]:
                clip_name = clip_names[now_clip_idx]
                # move images to dir use shutil
                save_dir_path = osp.join(save_path, clip_name)
                mkdir_or_exist(save_dir_path)
                # index from [total_deal, total_deal + do_frames)
                for idx in range(total_deal, total_deal + do_frames):
                    ensemble_path_1 = osp.join(save_path, "idx_{}_epoch_1.png".format(idx))
                    desti_path = osp.join(save_dir_path, str(idx - total_deal + start_index).zfill(padding_len) + ".png")
                    if osp.exists(ensemble_path_1):
                        # get the content
                        path = osp.join(save_path, "idx_{}.png".format(idx))
                        sum_result = imread(path, flag='unchanged').astype(np.float32)
                        os.remove(path)
                        for e in range(1, 8):
                            path = osp.join(save_path, "idx_{}_epoch_{}.png".format(idx, e))
                            sum_result = sum_result + imread(path, flag='unchanged').astype(np.float32)
                            os.remove(path)
                        sum_result = sum_result / 8
                        # 四舍五入
                        sum_result = sum_result.round().astype(np.uint8)
                        # save
                        imwrite(sum_result, desti_path)
                    else:
                        # move
                        shutil.move(osp.join(save_path, "idx_" + str(idx) + ".png"), desti_path)

                total_deal += do_frames
                do_frames = 0
                now_clip_idx += 1

    def evaluate(self, results, save_path):
        """ Evaluate with different metrics.
            Args:
                results (list of dict): for every dict, record metric -> value for one frame

            Return:
                dict: Evaluation results dict.
        """
        save_SVG_path = osp.join(save_path, "SVG")
        mkdir_or_exist(save_SVG_path)
        assert is_list_of(results, dict), f'results must be a list of dict, but got {type(results)}'
        assert len(results) >= len(self), "results length should >= dataset length, due to multicard eval"
        self.logger.info("eval samples length: {}, dataset length: {}, only select front {} results".format(len(results), len(self), len(self)))
        results = results[:len(self)]

        clip_names = sorted(self.frame_num.keys())  # e.g. [`city`, `walk`]
        frame_nums = [ self.frame_num[clip] for clip in clip_names ]

        eval_results = defaultdict(list)  # a dict of list
        
        do_frames = 0
        now_clip_idx = 0
        eval_results_one_clip = defaultdict(list)
        for res in results:
            for metric, val in res.items():
                eval_results_one_clip[metric].append(val)

            do_frames += 1
            if do_frames == frame_nums[now_clip_idx]: # 处理一个clip
                clip_name = clip_names[now_clip_idx]
                self.logger.info("{}: {} is ok".format(now_clip_idx, clip_name))
                for metric, values in eval_results_one_clip.items():
                    # metric clip_name values   to save an svg
                    average = sum(values) / len(values)
                    save_filename = clip_name + "_" + metric 
                    title = "{} for {}, length: {}, average: {:.4f}".format(metric, clip_name, len(values), average)
                    plt.figure(figsize=(len(values) // 4 + 1, 8))
                    plt.plot(list(range(len(values))), values, label=metric)  # promise that <= 10000
                    plt.title(title)
                    plt.xlabel('frame idx')
                    plt.ylabel('{} value'.format(metric))
                    plt.legend()
                    fig = plt.gcf()
                    fig.savefig(osp.join(save_SVG_path, save_filename + '.svg'), dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.clf()
                    plt.close()

                    eval_results[metric].append(average)

                do_frames = 0
                now_clip_idx += 1
                eval_results_one_clip = defaultdict(list)

        for metric, val_list in eval_results.items():
            assert len(val_list) == len(clip_names), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(clip_names)}')

        # average the results
        eval_results = {
            metric: sum(values) / len(values)
            for metric, values in eval_results.items()
        }

        return eval_results
