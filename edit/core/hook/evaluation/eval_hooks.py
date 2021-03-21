import os
import time
from megengine.distributed.group import is_distributed
import megengine.distributed as dist
from megengine.data.dataloader import DataLoader
from edit.core.hook import Hook
from edit.utils import to_list, is_list_of, get_logger, mkdir_or_exist


class EvalIterHook(Hook):
    """evaluation hook by iteration-based.

    This hook will regularly perform evaluation in a given interval

    Args:
        dataloader (DataLoader): A mge dataloader.
        interval (int): Evaluation interval. Default: 3000.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a mge DataLoader, but got {}'.format(type(dataloader)))
        self.dataloader = dataloader
        self.eval_kwargs = eval_kwargs
        self.interval = self.eval_kwargs.pop('interval', 10000)
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)
        self.log_path = self.eval_kwargs.pop('log_path', None)
        self.multi_process = self.eval_kwargs.pop('multi_process', False)
        self.ensemble = self.eval_kwargs.pop('ensemble', False)
        mkdir_or_exist(self.save_path)
        self.logger = get_logger(name = "EvalIterHook", log_file=self.log_path) # only for rank0
        
        if is_distributed():
            self.local_rank = dist.get_rank()
            self.nranks = dist.get_world_size()
        else:
            self.local_rank = 0
            self.nranks = 1

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        self.logger.info("start to eval for iter: {}".format(runner.iter+1))
        save_path = os.path.join(self.save_path, "iter_{}".format(runner.iter+1))
        mkdir_or_exist(save_path)
        results = []  # list of dict
        if self.multi_process:
            assert is_distributed(), "when set multiprocess eval, you should use multi process training"
            raise NotImplementedError("not support multi process for eval now")
        elif self.local_rank == 0:  # 全部交给rank0来处理
            for data in self.dataloader:
                outputs = runner.model.test_step(data, save_image=self.save_image, save_path=save_path, ensemble=self.ensemble)
                result = runner.model.cal_for_eval(outputs, data)
                assert isinstance(result, list)
                results += result
            self.evaluate(results, runner.iter+1)
        else:
            pass

        if is_distributed():
            dist.group_barrier()

    def evaluate(self, results, iters):
        """Evaluation function.

        Args:
            runner (``BaseRunner``): The runner.
            results (list of dict): Model forward results.
            iter: now iter.
        """
        save_path = os.path.join(self.save_path, "iter_{}".format(iters))  # save for some information. e.g. SVG for everyframe value in VSR.
        eval_res = self.dataloader.dataset.evaluate(results, save_path)
        self.logger.info("*****   eval results for {} iters:   *****".format(iters))
        for name, val in eval_res.items():
            self.logger.info("metric: {}  average_val: {:.4f}".format(name, val))
