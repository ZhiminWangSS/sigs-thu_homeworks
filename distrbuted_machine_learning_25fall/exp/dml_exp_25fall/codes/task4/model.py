import os
from urllib import parse
import torch
from torch.distributed.distributed_c10d import get_rank, get_world_size 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import wandb
import time

import argparse
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync
import dist_utils

class SubNetConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6,
                               kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)

    def forward(self, x_rref):
        """
        x_rref: RRef of input tensor (from worker0)
        """
        x = x_rref.to_here()  # 获取真实 tensor
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28→14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14→5
        return x

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class SubNetFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x_rref):
        x = x_rref.to_here()
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class ParallelNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        print(f"Initializing ParallelNet on worker {dist_utils.get_local_rank()}...")
        
        # 将两个子网络分别放置在 worker1 / worker2
        print(f"Creating remote SubNetConv on worker1...")
        self.conv_rref = rpc.remote("worker1", SubNetConv, args=(in_channels,))
        print(f"Creating remote SubNetFC on worker2...")
        self.fc_rref   = rpc.remote("worker2", SubNetFC, args=(num_classes,))
        print(f"ParallelNet initialization complete.")

    def forward(self, x):
        # 将 x 放入 RRef
        x_rref = rpc.RRef(x)

        # 调用远程卷积网络
        start_time = time.time()
        y_rref = self.conv_rref.rpc_async().forward(x_rref)
        y = y_rref.wait()
        end_time = time.time()
        comm_time = (end_time - start_time) * 1000 # ms
        data_size = (x.nelement() * x.element_size()) / (1024 * 1024) # MB
        wandb.log({"forward_conv_comm_time_ms": comm_time, "forward_conv_data_size_mb": data_size, "forward_conv_bandwidth_mb_s": data_size / (comm_time / 1000) if comm_time > 0 else 0})

        # 调用远程全连接网络
        start_time = time.time()
        out_rref = self.fc_rref.rpc_async().forward(rpc.RRef(y))
        out = out_rref.wait()
        end_time = time.time()
        comm_time = (end_time - start_time) * 1000 # ms
        data_size = (y.nelement() * y.element_size()) / (1024 * 1024) # MB
        wandb.log({"forward_fc_comm_time_ms": comm_time, "forward_fc_data_size_mb": data_size, "forward_fc_bandwidth_mb_s": data_size / (comm_time / 1000) if comm_time > 0 else 0})
        return out

    def parameter_rrefs(self):
        """Fetch remote parameters for DistributedOptimizer."""
        params = []
        # 使用 rpc_sync 从远端 SubNet 对象上同步调用 parameter_rrefs()
        params.extend(self.conv_rref.rpc_sync().parameter_rrefs())
        params.extend(self.fc_rref.rpc_sync().parameter_rrefs())
        return params

def train(model, dataloader, loss_fn, optimizer, num_epochs=2):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    model.train()
    dist_utils.init_parameters(model)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # 1. 创建一个分布式 autograd 上下文
            with dist_autograd.context() as context_id:

                # 2. 远程前向
                outputs = model(inputs)

                # 3. 损失
                loss = loss_fn(outputs, labels)

                # 4. 反向传播（跨机器）
                start_time = time.time()
                dist_autograd.backward(context_id, [loss])
                end_time = time.time()
                comm_time = (end_time - start_time) * 1000 # ms
                # Approximate data size (gradients)
                data_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024) # MB
                wandb.log({"backward_comm_time_ms": comm_time, "backward_data_size_mb": data_size, "backward_bandwidth_mb_s": data_size / (comm_time / 1000) if comm_time > 0 else 0})

                # 5. 更新分布式参数
                start_time = time.time()
                optimizer.step(context_id)
                end_time = time.time()
                comm_time = (end_time - start_time) * 1000 # ms
                data_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024) # MB
                wandb.log({"optimizer_comm_time_ms": comm_time, "optimizer_data_size_mb": data_size, "optimizer_bandwidth_mb_s": data_size / (comm_time / 1000) if comm_time > 0 else 0})

            if i % 100 == 0:
                print(f"[Epoch {epoch}] step {i}, loss = {loss.item():.4f}")
                wandb.log({"loss": loss.item()})

    print("Training Finished!")


def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))


def main():
    args = parse_args()
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    DATA_PATH = "./data"
    if args.rank == 0:
        wandb.init(entity="zhiminwang", project="dml_exp_25fall_task4", name=f"rank_{args.rank}")
        
        rpc.init_rpc("worker0", rank=args.rank, world_size=args.n_devices)
        # construct the model
        model = ParallelNet(in_channels=1, num_classes=10)
        # construct the dataset
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

        # construct the loss_fn and optimizer
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        dist_optimizer = DistributedOptimizer(torch.optim.SGD, model.parameter_rrefs(), lr=0.01)

        train(model, train_loader, loss_fn, dist_optimizer)
        test(model, test_loader)
        print("Progress start on the worker0...")
    
    elif args.rank == 1:
        rpc.init_rpc("worker1", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker1...")

    elif args.rank == 2:
        rpc.init_rpc("worker2", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker2...")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()