#!/bin/bash
# Adam优化器 - 固定beta1=0.9/beta2=0.999，调整学习率
for lr in 0.1 0.01 0.001 0.0001; do
    python run_all.py --optimizer Adam \
                      --lr ${lr}  \
                      --beta1 0.9 \
                      --beta2 0.999 \
                      --run_name "adam_${lr}_0.9_0.999"
done