import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def validate_distributed_config(opt):
    if getattr(opt, 'num_gpus', 1) < 1:
        raise ValueError('--num_gpus must be at least 1.')

    if opt.gpu_id == -1:
        if opt.num_gpus != 1:
            raise ValueError('CPU training only supports --num_gpus 1.')
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError('CUDA is not available, but GPU training was requested.')

    max_gpu_index = opt.gpu_id + opt.num_gpus - 1
    if opt.gpu_id < 0 or max_gpu_index >= available_gpus:
        raise ValueError(
            f'Requested GPUs [{opt.gpu_id}, {max_gpu_index}] but only {available_gpus} CUDA device(s) are available.'
        )

    if opt.batch_size < opt.num_gpus:
        raise ValueError('--batch_size must be >= --num_gpus so each process receives at least one sample.')

    if opt.batch_size % opt.num_gpus != 0:
        raise ValueError('--batch_size must be divisible by --num_gpus. The script treats batch_size as the global batch size.')


def is_distributed(opt):
    return getattr(opt, 'distributed', False)


def is_main_process(opt):
    return getattr(opt, 'rank', 0) == 0


def get_backend():
    if torch.cuda.is_available() and os.name != 'nt':
        return 'nccl'
    return 'gloo'


def setup_distributed(rank, opt):
    opt.rank = rank
    opt.world_size = opt.num_gpus if opt.gpu_id != -1 else 1
    opt.distributed = opt.world_size > 1
    opt.is_main_process = rank == 0

    if opt.gpu_id == -1:
        opt.local_gpu_id = -1
        opt.device = torch.device('cpu')
        return

    opt.local_gpu_id = opt.gpu_id + rank
    opt.device = torch.device(f'cuda:{opt.local_gpu_id}')
    torch.cuda.set_device(opt.local_gpu_id)

    if opt.distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(opt.master_port)
        dist.init_process_group(
            backend=get_backend(),
            rank=rank,
            world_size=opt.world_size,
        )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def wrap_model_for_distributed(model, opt):
    model = model.to(opt.device)
    if is_distributed(opt):
        return DDP(
            model,
            device_ids=[opt.local_gpu_id] if opt.gpu_id != -1 else None,
            output_device=opt.local_gpu_id if opt.gpu_id != -1 else None,
            broadcast_buffers=False,
        )
    return model


def per_process_batch_size(opt):
    if is_distributed(opt):
        return opt.batch_size // opt.world_size
    return opt.batch_size


def build_dataloader(dataset, opt, shuffle, drop_last, pin_memory=True):
    sampler = None
    if is_distributed(opt):
        sampler = DistributedSampler(
            dataset,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=per_process_batch_size(opt),
        sampler=sampler,
        shuffle=sampler is None and shuffle,
        drop_last=drop_last,
        num_workers=getattr(opt, 'num_workers', 4),
        pin_memory=pin_memory and opt.gpu_id != -1,
    )


def set_epoch_for_sampler(dataloader, epoch):
    sampler = getattr(dataloader, 'sampler', None)
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)


def reduce_mean(value, device):
    if torch.is_tensor(value):
        tensor = value.detach().to(device)
        if tensor.ndim != 0:
            tensor = tensor.mean()
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def launch_training(opt, worker_fn):
    validate_distributed_config(opt)

    if opt.gpu_id == -1 or opt.num_gpus == 1:
        worker_fn(0, copy.deepcopy(opt))
        return

    mp.spawn(worker_fn, nprocs=opt.num_gpus, args=(copy.deepcopy(opt),), join=True)
