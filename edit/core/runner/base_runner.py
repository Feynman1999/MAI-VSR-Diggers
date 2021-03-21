import os.path as osp
from abc import ABCMeta, abstractmethod
import megengine as mge
import megengine.distributed as dist
from megengine.optimizer.optimizer import Optimizer
from megengine.module import Module
from edit.utils import mkdir_or_exist, build_from_cfg, get_root_logger
from ..hook import Hook, HOOKS, get_priority

module_ckpt_suffix = "_module.mge"
optim_ckpt_suffix = "_optim.mge"

class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for Mge.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``test()``
    - ``save_checkpoint()``
    - ``resume()``

    Args:
        model (:obj:`megengine.module.Module`): The model to be run.
        optimizers_cfg (dict): optimizer configs
        work_dir (str, optional): The working directory to save checkpoints and logs. Defaults to None.
    """

    def __init__(self, model, optimizers_cfg=None, work_dir=None):
        assert hasattr(model, 'train_step')
        assert hasattr(model, 'test_step')
        assert hasattr(model, 'create_gradmanager_and_optimizers')
        assert hasattr(model, 'cal_for_eval')

        self.model = model
        self.optimizers_cfg = optimizers_cfg
        self.logger = get_root_logger()
        self.work_dir = work_dir
        assert self.work_dir is not None

        # get model name from the model class
        self._model_name = self.model.__class__.__name__
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self, data_loader):
        pass

    @abstractmethod
    def test(self, data_loader):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, max_iters):
        pass

    @abstractmethod
    def save_checkpoint(self, out_dir, create_symlink=True):
        pass

    @abstractmethod
    def resume(self, path2checkpoint):
        pass

    @abstractmethod
    def register_training_hooks(self, lr_config, checkpoint_config, log_config):
        """Register default hooks for training.

            Default hooks include:

            - LrUpdaterHook
            - CheckpointSaverHook
            - log_config
        """
        pass

    def create_gradmanager_and_optimizers(self):
        self.model.create_gradmanager_and_optimizers(self.optimizers_cfg)

    def sync_model_params(self):
        if dist.is_distributed():
            self.logger.info("syncing the model's parameters...")
            dist.bcast_list_(self.model.parameters(), dist.WORLD)
        else:
            pass  # do nothing

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        raise NotImplementedError("")
        # if isinstance(self.optimizer, Optimizer):
        #     lr = [group['lr'] for group in self.optimizer.param_groups]
        # elif isinstance(self.optimizer, dict):
        #     lr = dict()
        #     for name, optim in self.optimizer.items():
        #         lr[name] = [group['lr'] for group in optim.param_groups]
        # else:
        #     raise RuntimeError('lr is not applicable because optimizer does not exist.')
        # return lr

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        raise NotImplementedError("")
        # def _get_momentum(optimizer):
        #     momentums = []
        #     for group in optimizer.param_groups:
        #         if 'momentum' in group.keys():
        #             momentums.append(group['momentum'])
        #         elif 'betas' in group.keys():
        #             momentums.append(group['betas'][0])
        #         else:
        #             momentums.append(0)
        #     return momentums
        #
        # if self.optimizer is None:
        #     raise RuntimeError('momentum is not applicable because optimizer does not exist.')
        # elif isinstance(self.optimizer, Optimizer):
        #     momentums = _get_momentum(self.optimizer)
        # elif isinstance(self.optimizer, dict):
        #     momentums = dict()
        #     for name, optim in self.optimizer.items():
        #         momentums[name] = _get_momentum(optim)
        # return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hook')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, path2checkpoint, load_optim=True):
        """
            :param path2checkpoint: e.g. workdirs/xxxxx/checkpoint/epoch_10
            :return: dict
        """
        assert osp.exists(path2checkpoint), "{} do not exist".format(path2checkpoint)
        dirname = osp.split(path2checkpoint)[-1]
        epoch, nums = dirname.split("_")
        assert epoch in ("epoch", )
        self.logger.info('load checkpoint from {}'.format(path2checkpoint))
        # 遍历model中的所有配置optimizer的model，并进行load
        res = dict()
        res['nums'] = int(nums)
        for submodule_name in self.optimizers_cfg.keys():
            submodule = getattr(self.model, submodule_name, None)
            assert submodule is not None, "model should have submodule {}".format(submodule_name)
            assert isinstance(submodule, Module), "submodule should be instance of mge.module.Module"
            if dist.get_rank() == 0:
                module_state_dict = mge.load(osp.join(path2checkpoint, submodule_name + module_ckpt_suffix))
                submodule.load_state_dict(module_state_dict, strict = False)
            if load_optim:
                optim_state_dict = mge.load(osp.join(path2checkpoint, submodule_name + optim_ckpt_suffix))
                res[submodule_name] = optim_state_dict
        return res

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = build_from_cfg(info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='HIGH')
