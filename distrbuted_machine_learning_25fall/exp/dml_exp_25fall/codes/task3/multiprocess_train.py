import os
import torch
import torch.multiprocessing as mp
import argparse
from model import parse_args

def run_training(rank, world_size, args):
    """
    在每个进程中运行的训练函数
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 修改参数中的rank和world_size
    args.rank = rank
    args.n_devices = world_size
    
    # 导入并执行主训练函数
    from model import main
    main(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=4, type=int, help="Number of processes to spawn")
    parser.add_argument('--gpu', default="0,1", type=str, help='GPU IDs')
    parser.add_argument('--master_addr', default='localhost', type=str, help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str, help='port for distributed training')
    parser.add_argument('--sampler_type', default='mysampler', type=str, 
                       choices=['mysampler', 'randomsampler', 'randomsplitsampler'], 
                       help='Sampler type for data loading')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--log_dir', type=str, default='./logs', help='directory for tensorboard logs (default: ./logs)')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Starting multiprocess training with {args.n_devices} processes")
    print(f"Using sampler type: {args.sampler_type}")
    print(f"Using GPUs: {args.gpu}")
    
    # 使用spawn启动多进程
    mp.spawn(run_training,
             args=(args.n_devices, args),
             nprocs=args.n_devices,
             join=True)

if __name__ == "__main__":
    main()