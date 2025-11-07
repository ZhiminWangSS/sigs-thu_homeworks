#!/bin/bash

# 并行启动所有训练脚本
# 作者: AI助手
# 描述: 同时启动 Adam、Adam-MLP、SGDM 和 SGDM-MLP 训练

echo "开始并行启动所有训练脚本..."

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 启动 run_adam.py (CNN模型)
echo "启动 Adam (CNN) 训练..."
python run_adam_cnn.py &
ADAM_PID=$!

# 启动 run_adam_mlp.py (MLP模型)
echo "启动 Adam (MLP) 训练..."
python run_adam_mlp.py &
ADAM_MLP_PID=$!

# 启动 run_sgdm.py (CNN模型)
echo "启动 SGDM (CNN) 训练..."
python run_sgdm_cnn.py &
SGDM_PID=$!

# 启动 run_sgdm_mlp.py (MLP模型)
echo "启动 SGDM (MLP) 训练..."
python run_sgdm_mlp.py &
SGDM_MLP_PID=$!

echo "所有训练脚本已启动，进程ID如下："
echo "Adam (CNN): $ADAM_PID"
echo "Adam (MLP): $ADAM_MLP_PID"
echo "SGDM (CNN): $SGDM_PID"
echo "SGDM (MLP): $SGDM_MLP_PID"

echo ""
echo "训练正在进行中..."
echo "可以使用以下命令查看进程状态："
echo "ps -p $ADAM_PID,$ADAM_MLP_PID,$SGDM_PID,$SGDM_MLP_PID"
echo ""
echo "等待所有训练完成..."

# 等待所有进程完成
wait $ADAM_PID
ADAM_EXIT=$?

wait $ADAM_MLP_PID
ADAM_MLP_EXIT=$?

wait $SGDM_PID
SGDM_EXIT=$?

wait $SGDM_MLP_PID
SGDM_MLP_EXIT=$?

echo ""
echo "所有训练已完成！退出状态如下："
echo "Adam (CNN): $ADAM_EXIT"
echo "Adam (MLP): $ADAM_MLP_EXIT"
echo "SGDM (CNN): $SGDM_EXIT"
echo "SGDM (MLP): $SGDM_MLP_EXIT"

# 检查是否有失败的进程
if [ $ADAM_EXIT -eq 0 ] && [ $ADAM_MLP_EXIT -eq 0 ] && [ $SGDM_EXIT -eq 0 ] && [ $SGDM_MLP_EXIT -eq 0 ]; then
    echo "所有训练都成功完成！"
else
    echo "部分训练出现错误，请检查日志文件。"
fi

echo ""
echo "TensorBoard 日志保存在 logs/ 目录下"
echo "可以使用以下命令启动 TensorBoard："
echo "tensorboard --logdir logs/"