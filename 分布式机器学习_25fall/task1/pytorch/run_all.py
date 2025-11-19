#!/usr/bin/env python3
"""
综合启动脚本，支持命令行参数选择优化器
使用方法:
    python run_all.py --optimizer SGD --lr 0.01 --run_name my_experiment
    python run_all.py --optimizer SGDM --lr 0.01 --beta1 0.9
    python run_all.py --optimizer Adam --lr 0.001 --beta1 0.9 --beta2 0.999
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import main

def parse_args():
    parser = argparse.ArgumentParser(description='训练模型并选择优化器')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                       choices=['SGD', 'SGDM', 'Adam'],
                       help='优化器类型 (默认: SGD)')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率 (默认: SGD/SGDM=0.01, Adam=0.001)')
    parser.add_argument('--beta1', type=float, default=None,
                       help='一阶动量衰减率 (默认: 0.9)')
    parser.add_argument('--beta2', type=float, default=None,
                       help='二阶动量衰减率 (默认: 0.999)')
    parser.add_argument('--run_name', type=str, default=None,
                       help='TensorBoard运行名称 (默认: 自动生成)')
    parser.add_argument('--profiler', action='store_true', help='启用TensorBoard Profiler')
    
    return parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()
    
    print(f"使用优化器: {args.optimizer}")
    if args.lr:
        print(f"学习率: {args.lr}")
    if args.beta1:
        print(f"beta1: {args.beta1}")
    if args.beta2:
        print(f"beta2: {args.beta2}")
    if args.profiler:
        print("启用TensorBoard Profiler")
    
    main(
        optimizer_type=args.optimizer,
        run_name=args.run_name,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        use_profiler=args.profiler
    )