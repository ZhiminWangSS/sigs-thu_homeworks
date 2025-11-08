#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================
# HWXOR1_SYNC.PY —— 与 Dr. ZhuoQing HWXOR1 结构完全同步版
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 样本构造
xor_x0 = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])  # 输入样本
xor_y0 = np.array([-1, 1, 1, -1]).reshape(1, -1)          # 输出样本

# ------------------------------------------------------------
# 双曲正切激活函数及导数
def tanh(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))  # 手写tanh形式

def tanh_deriv(x):
    return 1 - np.power(x, 2)  # x为tanh输出时导数形式

# ------------------------------------------------------------
# 初始化权重
def initialize_parameters():
    random.seed(2)
    parameters = {
        'w10': random.uniform(-0.5, 0.5),
        'w20': random.uniform(-0.5, 0.5),
        'w13': random.uniform(-0.5, 0.5),
        'w12': random.uniform(-0.5, 0.5),
        'w14': random.uniform(-0.5, 0.5),
        'w23': random.uniform(-0.5, 0.5),
        'w24': random.uniform(-0.5, 0.5),
    }
    return parameters

# ------------------------------------------------------------
# 前向传播
def forward_propagate(X, parameters):
    w10 = parameters['w10']
    w20 = parameters['w20']
    w13 = parameters['w13']
    w12 = parameters['w12']
    w14 = parameters['w14']
    w23 = parameters['w23']
    w24 = parameters['w24']

    W2 = np.array([w23, w24])  # 输入→隐层权重
    W1 = np.array([w13, w14])  # 输入→输出层权重
    Z2 = np.dot(W2.T, X.T) - w20  # 隐层净输入
    A2 = tanh(Z2)                # 隐层输出

    Z1 = np.dot(W1.T, X.T) + w12 * A2 - w10
    A1 = Z1                      # 输出层为线性输出（原始版本）
    # 若希望非线性输出，可改为 A1 = tanh(Z1)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A1, cache

# ------------------------------------------------------------
# 代价函数
def calculate_cost(A1, Y):
    err = A1 - Y
    cost = np.dot(err, err.T) / Y.shape[1]
    return cost

# ------------------------------------------------------------
# 反向传播
def backward_propagate(parameters, cache, X, Y):
    m = X.shape[0]
    w10 = parameters['w10']
    w20 = parameters['w20']
    w13 = parameters['w13']
    w12 = parameters['w12']
    w14 = parameters['w14']
    w23 = parameters['w23']
    w24 = parameters['w24']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ1 = A1 - Y
    d10 = -1 * np.sum(dZ1, axis=1, keepdims=True) / m
    d13 = np.dot(dZ1, X.T[0].T) / m
    d12 = np.dot(dZ1, A2.T) / m
    d14 = np.dot(dZ1, X.T[1].T) / m

    dZ2 = w12 * dZ1 * (1 - np.power(A2, 2))
    d23 = np.dot(dZ2, X.T[0].T) / m
    d24 = np.dot(dZ2, X.T[1].T) / m
    d20 = -1 * np.sum(dZ2, axis=1, keepdims=True) / m

    grads = {'d10': d10, 'd20': d20, 'd13': d13,
             'd12': d12, 'd14': d14, 'd23': d23, 'd24': d24}
    return grads

# ------------------------------------------------------------
# 参数更新
def update_parameters(parameters, grads, lr):
    for key in ['w10', 'w20', 'w13', 'w12', 'w14', 'w23', 'w24']:
        grad_value = grads['d' + key[1:]]
        # 如果梯度是数组，取其第一个元素（标量值）
        if hasattr(grad_value, 'shape') and grad_value.shape:
            grad_value = grad_value.item() if grad_value.size == 1 else grad_value[0, 0]
        parameters[key] -= lr * grad_value
    return parameters

# ------------------------------------------------------------
# 训练函数
def train(X, Y, num_iterations, lr=0.5):
    parameters = initialize_parameters()
    costdim = []

    for i in range(num_iterations):
        A1, cache = forward_propagate(X, parameters)
        cost = calculate_cost(A1, Y)
        grads = backward_propagate(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, lr)

        if i % 50 == 0:
            print(f"Iteration {i}, Cost = {float(cost):.6f}")
            costdim.append(float(cost))
            if cost < 0.01:
                break
    return parameters, costdim

# ------------------------------------------------------------
# 执行训练并绘制结果
parameter, costdim = train(xor_x0, xor_y0, 1000, 0.5)

A1, cache = forward_propagate(xor_x0, parameter)
print("\n=== 训练结果 ===")
print("预测输出：", A1)
print("目标输出：", xor_y0)

plt.plot(np.arange(len(costdim)) * 50, costdim)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("xor_cost_curve_sync.png", dpi=200)

# ------------------------------------------------------------
# END OF FILE
# ==========================================================
