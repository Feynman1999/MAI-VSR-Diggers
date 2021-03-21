from megengine.data.dataset import Dataset
from .registry import DATASETS


@DATASETS.register_module()
class RepeatDataset(Dataset):
    """A wrapper of MapDataset dataset to repeat.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        super(RepeatDataset, self).__init__()
        self.dataset = dataset
        self.times = times
        dataset.logger.info("use repeatdataset, repeat times: {}".format(times))
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.times * self._ori_len
