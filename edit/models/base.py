from abc import abstractmethod
import megengine.module as M
from megengine.distributed.group import get_rank
from edit.core.optimizer import build_optimizers, build_gradmanagers


class BaseModel(M.Module):
    """Base model.

    All models should subclass it.
    All subclass should overwrite:

        ``init_weights``, supporting to initialize models.

        ``train_step``, supporting to train one step when training.

        ``test_step``, supporting to test(predict) one step for eval and test.

        ``cal_for_eval``, do some calculation for eval.
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.local_rank = get_rank()

    def forward(self, *inputs, **kwargs):
        pass
    
    @abstractmethod
    def init_weights(self):
        """Abstract method for initializing weight.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def train_step(self, batchdata):
        """Abstract method for one training step.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def test_step(self, batchdata, **kwargs):
        """Abstract method for one test step.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        pass

    def create_gradmanager_and_optimizers(self, optimizers_cfg):
        """build gradmanager and optimizers use optimizers_cfg"""
        # check
        for _, cfg in optimizers_cfg.items():
            if not isinstance(cfg, dict):
                raise RuntimeError("please use 'dict of dict' style for optimizers config (used in grad manager too)")
        
        self.gms = build_gradmanagers(self, optimizers_cfg)
        self.optimizers = build_optimizers(self, optimizers_cfg)
