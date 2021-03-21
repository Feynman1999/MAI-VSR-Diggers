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
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF')


class BaseMatchDataset(BaseDataset):
    """Base class for matching dataset.
    """

    def __init__(self, pipeline, mode="train"):
        super(BaseMatchDataset, self).__init__(pipeline, mode)
        self.moving_average_len = 10
        self.pooling = defaultdict(list)  # key -> value list (less than or equal moving_average_len)
        self.difficulty_score = {1:0.4, 2:0.4, 3:0.4, 4:0.6, 5:0.6, 6:0.6, 7:0.8, 8:0.8}
        self.threshold = {1:5, 2:5, 3:5, 4:5, 5:5, 6:5, 7:5, 8:5}

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
        return self.pipeline(results)

    def evaluate(self, results, save_path):
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
            class_id = res['class_id']
            file_id = res['file_id']
            # 统计
            for metric, val in res.items():
                if "id" in metric:
                    continue 
                eval_results[metric].append(val)
                eval_results[metric+ "_" + str(class_id)].append(val)

                # 特殊打印
                if val > 5:
                    self.logger.info("{} value: {}".format(metric, val))
                    eval_results[metric+ "_" + str(class_id) + "_more_than_5_nums"].append(1.0)
                else:
                    # val = int(val+0.5)
                    integer = str(int(val))
                    point = str(int(val*10)%10)
                    eval_results[metric+ "_" + "{}.{}".format(integer, point) + "_nums"].append(1.0)
        
        # 根据eval_results[metric+ "_" + str(class_id)]计算分数信息
        def get_score_by_dis(x):
            if x>5:
                return 0
            else:
                return 6-x

        now_score = 0
        best_score = 0
        for class_id, diff_score in self.difficulty_score.items():
            key = "dis_"+str(class_id)
            if eval_results.get(key) is None:
                self.logger.info("do not have class index: {}".format(class_id))
            else:
                thre = self.threshold[class_id]
                list_0_1 = [ (dis_value<=thre) for dis_value in eval_results[key]]
                Lambda =  sum(list_0_1) * 1.0 / len(list_0_1)
                for dis_value in eval_results[key]:
                    now_score += get_score_by_dis(dis_value) * diff_score * Lambda
                    best_score += 6 * diff_score * 1.0
        now_score_percent = now_score*100/best_score
        self.logger.info("now competition score: {}".format(now_score_percent))

        ans = {}
        ans['competition_score'] = now_score_percent
        for metric, values in eval_results.items():
            if "nums" not in metric:
                ans[metric] = sum(values) / len(values)
            else:
                ans[metric] = sum(values)
            if metric == "dis":
                self.logger.info("now dis: {}".format(ans[metric]))

        # update pooling
        for metric, value in ans.items():
            self.pooling[metric].append(value)
            if len(self.pooling[metric]) > self.moving_average_len:
                # remove the first one
                self.pooling[metric].pop(0)

        # eval_results
        eval_results = {
            metric: sum(values) / len(values)
            for metric, values in self.pooling.items()
        }
        return eval_results
