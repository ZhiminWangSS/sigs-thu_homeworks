import os
import torch
import torch.distributed as dist 


def dist_init(world_size, rank, master_addr='localhost', master_port='12355'):
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
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
            dist.broadcast(param.data,0)


def allreduce_average_gradients(model):
    for param in model.parameters():
        # implement your own aggregation method
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

def allgather_average_gradients(model):
    world_size = get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            # Gather gradients from all processes
            gathered_grads = [torch.zeros_like(param.grad.data) for _ in range(world_size)]
            dist.all_gather(gathered_grads, param.grad.data)
            
            # Average the gathered gradients
            avg_grad = torch.stack(gathered_grads).mean(dim=0)
            param.grad.data = avg_grad


def reduce_average_gradients(model):
    """Average gradients using Reduce collective operation"""
    world_size = get_world_size()
    rank = get_local_rank()
    
    for param in model.parameters():
        if param.grad is not None:
            # Use reduce with AVG operation to rank 0, then broadcast back
            if rank == 0:
                # Rank 0 receives and averages gradients
                temp_grad = param.grad.data.clone()
                for src_rank in range(1, world_size):
                    dist.recv(temp_grad, src=src_rank)
                    param.grad.data += temp_grad
                param.grad.data /= world_size
            else:
                # Other ranks send their gradients to rank 0
                dist.send(param.grad.data, dst=0)
            
            # Broadcast the averaged gradients from rank 0 to all processes
            dist.broadcast(param.grad.data, src=0)


def allreduce_sum_gradients(model):
    """Sum gradients using AllReduce collective operation"""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


def destroy_process_group():
    """Clean up the distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()