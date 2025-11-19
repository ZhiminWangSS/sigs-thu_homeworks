#!/bin/bash
# SGDM优化器 - 固定学习率=0.01，调整beta1
for beta1 in 0.5 0.9 0.99; do
    python run_all.py --optimizer SGDM \
                      --lr 0.01 \
                      --beta1 ${beta1}  \
                      --run_name "sgdm_0.01_${beta1}"
done