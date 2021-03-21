import os
from megengine.distributed.group import get_rank, get_world_size
from edit.core.hook import Hook, HOOKS


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
    """

    def __init__(self, interval=-1, by_epoch=True, out_dir=None):
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = out_dir
        self.local_rank = get_rank()

    def after_train_epoch(self, runner):
        if self.local_rank > 0:
            return

        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        
        if not self.out_dir:
            self.out_dir = os.path.join(runner.work_dir, "checkpoints")
        
        runner.save_checkpoint(self.out_dir)

    def after_train_iter(self, runner):
        if self.local_rank > 0:
            return

        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = os.path.join(runner.work_dir, "checkpoints")
        
        runner.save_checkpoint(self.out_dir)
