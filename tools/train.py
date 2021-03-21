"""
    train your model and support eval when training.
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
    parser = argparse.ArgumentParser(description='Train and Eval an editor o(*￣▽￣*)ブ')
    parser.add_argument('config', help='train config file path')
    parser.add_argument("-d", "--dynamic", default=True, action='store_true', help="enable dygraph mode")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--gpuids", type=str, default="-1", help="spcefic gpus, -1 for cpu, >=0 for gpu, e.g.: 2,3")
    parser.add_argument('--work_dir', type=str, default=None, help='the dir to save logs and models')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    args = parser.parse_args()
    return args

def get_loader(dataset, cfg, mode='train'):
    assert mode in ('train', 'eval')
    if mode == 'train':
        sampler = RandomSampler(dataset, batch_size=cfg.data.samples_per_gpu, drop_last=True, seed=0)
        loader = DataLoader(dataset, sampler, num_workers=cfg.data.workers_per_gpu)
    else:
        samples_per_gpu = cfg.data.get('eval_samples_per_gpu', cfg.data.samples_per_gpu)
        workers_per_gpu = cfg.data.get('eval_workers_per_gpu', cfg.data.workers_per_gpu)
        if cfg.evaluation.multi_process is True:
            sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False)
        else:
            sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False, world_size=1, rank=0)
        loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    return loader

def train(model, datasets, cfg, rank):
    data_loaders = [ get_loader(ds, cfg, 'train') for ds in datasets]
    runner = EpochBasedRunner(model=model, optimizers_cfg=cfg.optimizers, work_dir=cfg.work_dir)
    
    runner.create_gradmanager_and_optimizers()  # 每个进程均创建gm和optimizers, 均是model的属性

    if cfg.resume_from is not None:
        # 恢复之前的训练,即epoch数目（包括模型参数和优化器）。若多卡训练则只有rank 0进程对模型加载参数（后面会同步）。如果resume optim，则每个进程均会load optim state.
        runner.resume(cfg.resume_from, cfg.get('resume_optim', True)) 
    elif cfg.load_from is not None:
        # 加载参数，但假装从头开始训练。若多卡训练则只有rank 0进程对模型加载参数 （后面会同步）。
        runner.load_checkpoint(cfg.load_from, load_optim=False) 
    else:
        pass  # 不加载任何参数，从头训练
    
    # 对模型参数进行同步
    runner.sync_model_params()

    # register some useful hooks
    runner.register_training_hooks(lr_config=cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # register evaluation hook
    if cfg.get('evaluation', None) is not None:
        dataset = build_dataset(cfg.data.eval)
        save_path = os.path.join(cfg.work_dir, 'eval_visuals')
        log_path = os.path.join(cfg.work_dir, 'eval.log')
        runner.register_hook(EvalIterHook(get_loader(dataset, cfg, 'eval'),  save_path=save_path, log_path=log_path, **cfg.evaluation))

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

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
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    model = build_model(cfg.model, train_cfg=cfg.train_cfg, eval_cfg=cfg.eval_cfg) # 此时参数已经随机化完成
    datasets = [build_dataset(cfg.data.train)]
    train(model, datasets, cfg, rank)

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.dynamic = args.dynamic
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        assert cfg.get('work_dir', None) is not None, 'if do not set work_dir in args, please set in config file'
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    cfg.work_dir = os.path.join(cfg.work_dir, timestamp)
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    log_file = os.path.join(cfg.work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('Config:\n{}'.format(cfg.text))
    
    gpu_list = [ item.strip() for item in args.gpuids.split(",")]
    if gpu_list[0] == "-1":
        world_size = 0 # use cpu
        logger.info('training use only cpu')
    else:
        world_size = len(gpu_list)
        logger.info('training gpus num: {}'.format(world_size))

    if world_size == 0: # use cpu
        mge.set_default_device(device='cpux')
    elif world_size == 1:
        mge.set_default_device(device='gpu' + gpu_list[0])
    else:
        pass

    if world_size > 1:
        # scale weight decay in "SUM" mode
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
