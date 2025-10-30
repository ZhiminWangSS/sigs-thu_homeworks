#!/bin/bash
# Adam优化器 - 固定学习率=0.001，调整beta1和beta2
for beta1 in 0.5 0.9 0.99; do
    for beta2 in 0.5 0.9 0.999; do
        python run_all.py --optimizer Adam \
                          --lr 0.001 \
                          --beta1 ${beta1}  \
                          --beta2 ${beta2}  \
                          --run_name "adam_0.001_${beta1}_${beta2}"
    done
done