from ..hook import Hook
from abc import ABCMeta, abstractmethod


class LoggerHook(Hook, metaclass=ABCMeta):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch if less than `interval`.
        by_epoch (bool): Whether EpochBasedRunner is used.
    """
    def __init__(self, interval=10, ignore_last=True, by_epoch=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.by_epoch = by_epoch

    @abstractmethod
    def log(self, runner):
        pass

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            self.log(runner)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            self.log(runner)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            self.log(runner)
        else:
            pass

    def after_test_epoch(self, runner):
        self.log(runner)
