#!/bin/bash
# SGDM优化器 - 固定beta1=0.9，调整学习率
for lr in 0.1 0.01 0.001 0.0001; do
    python run_all.py --optimizer SGDM \
                      --lr ${lr}  \
                      --beta1 0.9 \
                      --run_name "sgdm_${lr}_0.9"
done