#!/usr/bin/env python3
"""
SGDM优化器启动脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import main

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 使用SGDM优化器
    main(
        optimizer_type="SGDM",
        run_name="sgdm_experiment",
        lr=0.01,
        beta1=0.9
    )