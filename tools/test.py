"""
    test your model.
"""
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import multiprocessing as mp
import time
import argparse
import megengine as mge
import megengine.distributed as dist
from megengine.jit import trace
from megengine.data import RandomSampler, SequentialSampler, DataLoader

from edit.utils import Config, mkdir_or_exist, build_from_cfg, get_root_logger
from edit.models import build_model
from edit.datasets import build_dataset
from edit.core.runner import EpochBasedRunner
from edit.core.hook import HOOKS
from edit.core.hook.evaluation import EvalIterHook

def parse_args():
    parser = argparse.ArgumentParser(description='Test an editor o(*￣▽￣*)ブ')
    parser.add_argument('config', help='test config file path')
    parser.add_argument("-d", "--dynamic", default=True, action='store_true', help="enable dygraph mode")
    parser.add_argument("-e", "--ensemble", default=False, action = 'store_true')
    parser.add_argument("--gpuids", type=str, default="-1", help="spcefic gpus, -1 for cpu, >=0 for gpu, e.g.: 2,3")
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def get_loader(dataset, cfg):
    samples_per_gpu = cfg.data.test_samples_per_gpu
    workers_per_gpu = cfg.data.test_workers_per_gpu
    sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False)
    loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    return loader

def test(model, datasets, cfg, rank):
    data_loaders = [ get_loader(ds, cfg) for ds in datasets ]
    runner = EpochBasedRunner(model=model, optimizers_cfg=cfg.optimizers, work_dir=cfg.work_dir)
    runner.load_checkpoint(cfg.load_from, load_optim=False)
    runner.run(data_loaders, cfg.workflow, 1)

def worker(rank, world_size, cfg, gpu_id="0", port=23333):
    if cfg.dynamic:
        trace.enabled = False

    if world_size > 1:
        dist.init_process_group(
            master_ip = "localhost",
            port = port,
            world_size = world_size,
            rank = rank,
            device = int(gpu_id)%10,
        )
        log_file = os.path.join(cfg.work_dir, 'rank{}_root.log'.format(rank))
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)  # 给每个进程创立自己的root logger，但只有rank0的创建文件，其余的不创建且为error级别
    model = build_model(cfg.model, eval_cfg=cfg.eval_cfg)  # eval cfg can provide some useful info, e.g. the padding multi
    datasets = [build_dataset(cfg.data.test)]
    test(model, datasets, cfg, rank)

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.dynamic = args.dynamic
    cfg.ensemble = args.ensemble
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        assert cfg.get('work_dir', None) is not None, 'if do not set work_dir in args, please set in config file'

    cfg.work_dir = os.path.join(cfg.work_dir, timestamp)
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    log_file = os.path.join(cfg.work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('Config:\n{}'.format(cfg.text))

    gpu_list = [ item.strip() for item in args.gpuids.split(",")]
    if gpu_list[0] == "-1":
        world_size = 0 # use cpu
        logger.info('test use only cpu')
    else:
        world_size = len(gpu_list)
        logger.info('test gpus num: {}'.format(world_size))

    # assert world_size <= mge.get_device_count("gpu")

    if world_size == 0: # use cpu    
        mge.set_default_device(device='cpux')
    elif world_size == 1:
        mge.set_default_device(device='gpu' + gpu_list[0])
    else:
        pass
    
    if world_size > 1:
        port = dist.util.get_free_ports(1)[0]
        server = dist.Server(port)
        processes = []
        for rank in range(world_size):
            logger.info("init distributed process group {} / {}".format(rank, world_size))
            p = mp.Process(target=worker, args=(rank, world_size, cfg, gpu_list[rank], port))
            p.start()
            processes.append(p)

        for rank in range(world_size):
            processes[rank].join()
            code = processes[rank].exitcode
            assert code == 0, "subprocess {} exit with code {}".format(rank, code)
    else:
        worker(0, 1, cfg)

if __name__ == "__main__":
    main()
