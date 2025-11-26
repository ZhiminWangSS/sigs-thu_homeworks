import os
import torch
import torch.distributed as dist 
import time
import wandb


def dist_init(world_size, rank, master_addr='localhost', master_port='12355'):
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"Init the distributed environment on worker {rank}...")
    # initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    assert dist.is_initialized(), "Error! The distributed env is not initialized!"

    return True


def get_local_rank():
    # get the local rank (devices id)
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def init_parameters(model):
    # Boradcast the initial gradients of the model parameters
    if get_world_size() > 1:
        for param in model.parameters():
            start_time = time.time()
            dist.broadcast(param.data,0)
            end_time = time.time()
            comm_time = (end_time - start_time) * 1000 # ms
            data_size = (param.data.nelement() * param.data.element_size()) / (1024 * 1024) # MB
            if get_local_rank() == 0:
                wandb.log({"broadcast_comm_time_ms": comm_time, "broadcast_data_size_mb": data_size, "broadcast_bandwidth_mb_s": data_size / (comm_time / 1000) if comm_time > 0 else 0})


def average_gradients(model):
    # Aggregate the gradients on different devices, you can try other strategy.
    size = float(dist.get_world_size())
    # implement your own aggregation method
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size