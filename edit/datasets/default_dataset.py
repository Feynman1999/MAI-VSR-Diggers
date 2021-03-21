import os.path as osp
from pathlib import Path
from edit.utils import scandir
from .base_dataset import BaseDataset
from .registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')

@DATASETS.register_module()
class DefaultDataset(BaseDataset):
    def __init__(self,
                 folder,
                 pipeline):
        super(DefaultDataset, self).__init__(pipeline)
        self.folder = str(folder)
        self.data_infos = self.load_annotations()

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

    def load_annotations(self):
        data_infos = []
        paths = self.scan_folder(self.folder)
        for path in paths:
            data_infos.append(dict(img_path=path))
        self.logger.info("DefaultDataset dataset load ok, len:{}".format(len(data_infos)))
        return data_infos
