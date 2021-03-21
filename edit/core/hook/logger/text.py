from collections import OrderedDict
from ..hook import HOOKS
from .base import LoggerHook
from megengine import Tensor
from edit.utils import is_list_of, AVERAGE_POOL


@HOOKS.register_module()
class TextLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and saved in json file.

    Args:
        by_epoch (bool): Whether EpochBasedRunner is used.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
    """

    def __init__(self, interval = 10, ignore_last = True, by_epoch = False, average_length = 10):
        super(TextLoggerHook, self).__init__(interval, ignore_last, by_epoch)
        self.pool = AVERAGE_POOL(average_length=average_length)

    def log(self, runner):
        log_dict = OrderedDict()
        log_dict['epoch'] = runner.epoch
        log_dict['losses'] = runner.losses
        if self.by_epoch:
            log_dict['iter'] = runner.inner_iter
        else:
            log_dict['iter'] = runner.iter
            
        # TODO: learning rate 信息
        # TODO: 内存使用信息(cpu, gpu)

        log_items = []
        for name, val in log_dict.items():
            if isinstance(val, Tensor) or is_list_of(val, Tensor):
                if isinstance(val, list):
                    val = [ float(item.item()) for item in val ]
                else:
                    val = float(val.item())
                aver_val = self.pool.update(name, val)
                
                if isinstance(val, list):
                    val = ", ".join([ "{:.5f}".format(item) for item in val ])
                    aver_val = ", ".join([ "{:.5f}".format(item) for item in aver_val ])
                else:
                    val = "{:.5f}".format(val)
                    aver_val = "{:.5f}".format(aver_val)
                log_items.append("{}: [{}], {}_ma: [{}]".format(name, val, name, aver_val))
            else:
                log_items.append(f'{name}: {val}')

        log_str = ',  '.join(log_items)
        runner.logger.info(log_str)
