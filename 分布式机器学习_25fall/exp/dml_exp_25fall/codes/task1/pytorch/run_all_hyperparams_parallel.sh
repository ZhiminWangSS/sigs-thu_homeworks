#!/bin/bash

# 并行执行所有超参数对比脚本

echo "开始并行执行所有超参数对比训练..."
echo "========================================"

# 给所有脚本执行权限
chmod +x sgdm_fixed_beta.sh sgdm_fixed_lr.sh adam_fixed_beta.sh adam_fixed_lr.sh

# 启动所有脚本在后台运行
echo "启动 SGDM 固定动量调整学习率..."
./sgdm_fixed_beta.sh &
PID1=$!

echo "启动 SGDM 固定学习率调整动量..."
./sgdm_fixed_lr.sh &
PID2=$!

echo "启动 Adam 固定动量调整学习率..."
./adam_fixed_beta.sh &
PID3=$!

echo "启动 Adam 固定学习率调整动量..."
./adam_fixed_lr.sh &
PID4=$!

# 等待所有后台进程完成
echo ""
echo "所有脚本已启动，等待完成..."
echo "========================================"

wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?
wait $PID3
STATUS3=$?
wait $PID4
STATUS4=$?

# 检查所有进程的退出状态
echo ""
echo "执行结果汇总:"
echo "========================================"
if [ $STATUS1 -eq 0 ]; then
    echo "✓ SGDM固定动量调整学习率 - 完成"
else
    echo "✗ SGDM固定动量调整学习率 - 失败 (退出码: $STATUS1)"
fi

if [ $STATUS2 -eq 0 ]; then
    echo "✓ SGDM固定学习率调整动量 - 完成"
else
    echo "✗ SGDM固定学习率调整动量 - 失败 (退出码: $STATUS2)"
fi

if [ $STATUS3 -eq 0 ]; then
    echo "✓ Adam固定动量调整学习率 - 完成"
else
    echo "✗ Adam固定动量调整学习率 - 失败 (退出码: $STATUS3)"
fi

if [ $STATUS4 -eq 0 ]; then
    echo "✓ Adam固定学习率调整动量 - 完成"
else
    echo "✗ Adam固定学习率调整动量 - 失败 (退出码: $STATUS4)"
fi

echo ""
echo "所有超参数对比训练执行完毕！"
echo "可以在TensorBoard中查看所有训练结果:"
echo "tensorboard --logdir logs"