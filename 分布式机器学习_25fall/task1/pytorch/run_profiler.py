#!/usr/bin/env python3
"""
TensorBoard Profiler示例脚本
用于启用TensorBoard Profiler功能进行性能分析
"""

import argparse
from model import main

def parse_args():
    parser = argparse.ArgumentParser(description='TensorBoard Profiler示例')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                       choices=['SGD', 'SGDM', 'Adam'],
                       help='优化器类型 (默认: SGD)')
    parser.add_argument('--lr', type=float, default=None, 
                       help='学习率 (默认: SGD/SGDM=0.01, Adam=0.001)')
    parser.add_argument('--beta1', type=float, default=None, 
                       help='一阶动量衰减率 (默认: 0.9)')
    parser.add_argument('--beta2', type=float, default=None, 
                       help='二阶动量衰减率 (仅Adam优化器使用)')
    parser.add_argument('--run_name', type=str, default=None, 
                       help='TensorBoard运行名称 (默认: 自动生成)')
    
    return parser.parse_args()

def main_with_profiler():
    args = parse_args()
    
    print("=" * 50)
    print("TensorBoard Profiler 示例")
    print("=" * 50)
    print(f"优化器类型: {args.optimizer}")
    if args.lr:
        print(f"学习率: {args.lr}")
    if args.beta1:
        print(f"beta1: {args.beta1}")
    if args.beta2:
        print(f"beta2: {args.beta2}")
    print("启用 TensorBoard Profiler")
    print("=" * 50)
    
    # 强制启用profiler
    main(
        optimizer_type=args.optimizer,
        run_name=args.run_name,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        use_profiler=True
    )

if __name__ == "__main__":
    main_with_profiler()