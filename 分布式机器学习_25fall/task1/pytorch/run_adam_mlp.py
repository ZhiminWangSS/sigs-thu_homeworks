#!/usr/bin/env python3
"""
Adam优化器启动脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import main

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 使用Adam优化器
    main(
        optimizer_type="Adam",
        run_name="adam_experiment",
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        model_type="MLP"
    )