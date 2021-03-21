import copy
from abc import ABCMeta, abstractmethod
from megengine.data.dataset import Dataset
from .pipelines.compose import Compose
from edit.utils import get_root_logger
from edit.utils.logger import logger_initialized


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for Dataset.

    All datasets should subclass it.
    All subclasses should overwrite:

    ``load_annotations``, supporting to load information and generate image lists.

    Args:
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): If True, the dataset will work in test mode. Otherwise, in train mode.
    """
    def __init__(self, pipeline, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ("train", "test", "eval")
        self.mode = mode
        self.pipeline = Compose(pipeline)
        self.logger = get_root_logger()

    @abstractmethod
    def load_annotations(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        """

    @abstractmethod
    def evaluate(self, results):
        """Abstract function for evaluate.

        All subclasses should overwrite this function
        """

    def prepare_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.prepare_data(idx)

