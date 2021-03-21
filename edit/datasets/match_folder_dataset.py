import os.path as osp
import numpy as np
import random
from .base_match_dataset import BaseMatchDataset
from .registry import DATASETS
import copy


@DATASETS.register_module()
class MatchFolderDataset(BaseMatchDataset):
    def __init__(self,
                 data_path,
                 opt_folder,  # folder
                 sar_folder,  # folder
                 file_list_name,  # txt file
                 pipeline,
                 mode='train',
                 x_size = 800,
                 z_size = 512,
                 balance_flag = "None"  # support None | uniform | test
                 ):
        super(MatchFolderDataset, self).__init__(pipeline, mode)
        self.data_path = str(data_path)
        self.opt_folder = osp.join(self.data_path, opt_folder)
        self.sar_folder = osp.join(self.data_path, sar_folder)
        self.file_list = osp.join(self.data_path, file_list_name)
        self.x_size = x_size
        self.z_size = z_size
        self.balance_flag = balance_flag
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        # 获得所有的成对信息，从file_list_name中
        f = open(self.file_list, 'r')
        opt_list = []
        sar_list = []
        top_left_x = []
        top_left_y = []
        for line in f.readlines():
            if line.strip() == "":
                continue
            infos = line.strip().split(" ")
            if self.mode == "test":
                if len(infos) == 1:
                    continue
                assert len(infos) == 3
                opt_list.append(infos[2])
                sar_list.append(infos[1])
            else:
                assert len(infos) == 4
                opt_list.append(infos[0])
                sar_list.append(infos[1])
                top_left_x.append(float(infos[3]))
                top_left_y.append(float(infos[2]))
        f.close()

        ban_dict=[(5,95), (5,19), (5,20), (5,10), (5 ,44), (5 ,9), (3 ,193), (3 ,61), (1 ,187), (1 ,185), (1 ,179), (1 ,30), (1 ,95), (1 ,59), (1 ,52), (1 ,94), (1 ,26),(1 ,24), (1 ,23), (1, 43), (1, 3), (1,82),(2,455), (6,375), (6,329)]
        
        for i in range(len(opt_list)):
            if self.mode == "train":
                assert opt_list[i][-4] == '.'
                aaa = int(opt_list[i][0])
                bbb = int(opt_list[i][:-4].split("_")[-1])
                if (aaa, bbb) in ban_dict:
                    continue
                data_infos.append(
                    dict(
                        opt_path=osp.join(self.opt_folder, opt_list[i]),
                        sar_path=osp.join(self.sar_folder, sar_list[i]),
                        bbox = np.array([top_left_x[i], 
                                         top_left_y[i], 
                                         top_left_x[i] + self.z_size - 1, 
                                         top_left_y[i] + self.z_size - 1]).astype(np.float32),
                        class_id = np.array(aaa).astype(np.int32),
                        file_id = np.array(bbb).astype(np.int32)
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        opt_path=osp.join(self.opt_folder, opt_list[i]),
                        sar_path=osp.join(self.sar_folder, sar_list[i]),
                        bbox = np.array([top_left_x[i], 
                                         top_left_y[i], 
                                         top_left_x[i] + self.z_size - 1, 
                                         top_left_y[i] + self.z_size - 1]).astype(np.float32),
                        class_id = int(opt_list[i][0]),
                        file_id = int(opt_list[i][:-4].split("_")[-1])
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        opt_path = osp.join(self.opt_folder, opt_list[i]),
                        sar_path = osp.join(self.sar_folder, sar_list[i]),
                        class_id = int(opt_list[i][0]),
                        file_id = int(opt_list[i][:-4].split("_")[-1])
                    )
                )
            else:
                raise NotImplementedError("not known mode: {}".format(self.mode))
        
        if self.mode == "train":
            # 按照类别进行分类，分多个列表
            ans = []
            for i in range(10):
                ans.append([])
            for item in data_infos:
                ans[item['class_id']].append(copy.deepcopy(item))
                ans[item['class_id']].append(copy.deepcopy(item))
                ans[item['class_id']].append(copy.deepcopy(item)) # 加足够的数量，避免数量不够

            if self.balance_flag == "test":
                test_distribute = [400, 400, 400, 600, 300, 600]  # 2500
                data_infos = []
                for i in range(len(test_distribute)):
                    data_infos = data_infos + random.sample(ans[i+1], test_distribute[i])
                assert len(data_infos) == sum(test_distribute)
                
            elif self.balance_flag == "uniform":
                uniform_distribute = [313, 313, 313, 314, 313, 314]  # 1880 same to None
                data_infos = []
                for i in range(len(uniform_distribute)):
                    data_infos = data_infos + random.sample(ans[i+1], uniform_distribute[i])
                assert len(data_infos) == sum(uniform_distribute)
            else:
                pass

        self.logger.info("MatchFolder dataset load ok, mode:{} len:{}".format(self.mode, len(data_infos)))
        return data_infos
