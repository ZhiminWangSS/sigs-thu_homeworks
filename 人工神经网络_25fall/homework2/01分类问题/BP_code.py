#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================
# BP_XOR_Structure1.py —— 对应图中结构1的BP算法实现
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Sigmoid 激活函数及其导数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

# ----------------------------------------------------------
# 构造输入与目标输出（XOR）
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([[0], [1], [1], [0]])

# ----------------------------------------------------------
# 初始化参数
np.random.seed(2)
eta = 0.5      # 学习率
epochs = 10000

# 输入层：x1, x2, bias(-1)
# 隐层：3个神经元 (2,3,4)
# 输出层：1个神经元 (1)
# 对应图中：w02,w03,w04,w52,w53,w54,w62,w63,w64,w21,w31,w41,w01
W_input_hidden = np.random.uniform(-1, 1, (3, 3))  # 3 hidden × (2 inputs + 1 bias)
W_hidden_output = np.random.uniform(-1, 1, (1, 4)) # 1 output × (3 hidden + 1 bias)

# ----------------------------------------------------------
# 前向传播
def forward(x):
    # 添加偏置输入 -1
    x_in = np.hstack((np.array([[-1]]), x.reshape(1, -1)))   # [ -1, x1, x2 ]
    net_h = np.dot(W_input_hidden, x_in.T)                   # shape (3,1)
    y_h = sigmoid(net_h)
    y_h_with_bias = np.vstack((np.array([[-1]]), y_h))       # [ -1, h1, h2, h3 ]
    net_o = np.dot(W_hidden_output, y_h_with_bias)
    y_o = sigmoid(net_o)
    return x_in, y_h_with_bias, y_o

# ----------------------------------------------------------
# 反向传播
def backward(x_in, y_h_with_bias, y_o, target):
    global W_input_hidden, W_hidden_output

    # 输出层误差信号 δ1
    delta_o = (target - y_o) * sigmoid_deriv(y_o)  # shape (1,1)

    # 隐层误差信号 δ2, δ3, δ4
    hidden_out = y_h_with_bias[1:]  # 去掉偏置
    delta_h = hidden_out * (1 - hidden_out) * (W_hidden_output[:, 1:].T @ delta_o)

    # 更新权值
    W_hidden_output += eta * delta_o * y_h_with_bias.T
    W_input_hidden += eta * delta_h * x_in

# ----------------------------------------------------------
# 训练过程
errors = []
for epoch in range(epochs):
    err_sum = 0.0
    for i in range(len(X)):
        x = X[i]
        t = Y[i]
        x_in, y_h_with_bias, y_o = forward(x)
        err_sum += np.sum((t - y_o) ** 2)
        backward(x_in, y_h_with_bias, y_o, t)
    errors.append(err_sum / len(X))
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}, MSE = {err_sum / len(X):.6f}")

# ----------------------------------------------------------
# 测试结果
print("\n=== 训练结果 ===")
for i in range(len(X)):
    _, _, y_pred = forward(X[i])
    print(f"输入 {X[i]} -> 输出 {y_pred[0,0]:.4f}")

# 绘制损失曲线
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Error for XOR (Structure 1)')
plt.grid(True)
plt.savefig('BP_training_error.png')
plt.close()
