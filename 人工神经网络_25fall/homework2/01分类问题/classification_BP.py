#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#============================================================
# ANN HW2 - BP三类分类完整实验版
#============================================================
import numpy as np
import matplotlib.pyplot as plt
import random

#------------------------------------------------------------
# 数据：取PDF第一组三类样本
x_data = np.array([
    [0.75, 1.0], [0.5, 0.75], [0.25, 0.0],   # 类别1
    [0.5, 0.0],  [0.0, 0.0],  [1.0, 0.75],   # 类别2
    [1.0, 1.0],  [0.5, 0.25], [0.75, 0.5]    # 类别3
])

y_data = np.array([
    [1, -1, -1], [1, -1, -1], [1, -1, -1],
    [-1, 1, -1], [-1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]
]).T

#------------------------------------------------------------
# 坐标归一化到 (-1,1)
x_data = 2 * x_data - 1

#------------------------------------------------------------
def shuffledata(X, Y):
    id = list(range(X.shape[0]))
    random.shuffle(id)
    return X[id], (Y.T[id]).T

#------------------------------------------------------------
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.5
    W2 = np.random.randn(n_y, n_h) * 0.5
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    return {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

#------------------------------------------------------------
def forward_propagate(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    Z1 = np.dot(W1, X.T) + b1
    A1 = 1 / (1 + np.exp(-Z1))             # Sigmoid
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)                       # Tanh
    return A2, {'A1':A1, 'A2':A2}

#------------------------------------------------------------
def calculate_cost(A2, Y):
    return np.mean(np.sum((A2 - Y)**2, axis=0))

#------------------------------------------------------------
def backward_propagate(params, cache, X, Y):
    m = X.shape[0]
    W2 = params['W2']
    A1, A2 = cache['A1'], cache['A2']

    dZ2 = (A2 - Y) * (1 - A2**2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}

#------------------------------------------------------------
def update_parameters(params, grads, lr):
    for key in ['W1','b1','W2','b2']:
        params[key] -= lr * grads['d'+key]
    return params

#------------------------------------------------------------
def train(X, Y, n_h=6, lr=0.5, num_iter=8000, disp_step=500):
    n_x, n_y = 2, 3
    params = initialize_parameters(n_x, n_h, n_y)
    costdim = []
    X, Y = shuffledata(X, Y)
    for i in range(num_iter):
        A2, cache = forward_propagate(X, params)
        cost = calculate_cost(A2, Y)
        grads = backward_propagate(params, cache, X, Y)
        params = update_parameters(params, grads, lr)
        if i % disp_step == 0:
            print(f"[h={n_h}] Iter {i} Cost={cost:.4f}")
            costdim.append(cost)
    return params, costdim

#------------------------------------------------------------
def test_accuracy(params, X, Y):
    A2, _ = forward_propagate(X, params)
    pred = np.where(A2 >= 0, 1, -1)
    acc = np.sum(np.all(pred == Y, axis=0)) / Y.shape[1]
    return acc * 100

#------------------------------------------------------------
# 不同隐层节点数实验
hidden_list = [2, 3, 4, 5, 6, 8, 10]
acc_list, cost_all = [], []

for h in hidden_list:
    params, costdim = train(x_data, y_data, n_h=h, lr=0.5)
    acc = test_accuracy(params, x_data, y_data)
    acc_list.append(acc)
    cost_all.append(costdim)
    print(f"隐层节点 {h} → 分类准确率: {acc:.1f}%")

#------------------------------------------------------------
# 可视化：不同隐层节点准确率
plt.figure(figsize=(6,4))
plt.plot(hidden_list, acc_list, marker='o')
plt.title("隐层节点数 vs 分类准确率")
plt.xlabel("隐层节点数")
plt.ylabel("准确率(%)")
plt.grid(True)
plt.show()

#------------------------------------------------------------
# 修正版：噪声数据生成与训练
def add_noise_with_label(X, Y, n=20, noise_range=0.1):
    X_aug, Y_aug = [], []
    for i in range(len(X)):
        for _ in range(n):
            delta = np.random.uniform(-noise_range, noise_range, X[i].shape)
            X_aug.append(X[i] + delta)
            Y_aug.append(Y[:, i])  # 同步标签
    return np.array(X_aug), np.array(Y_aug).T

# 在 (-1,1) 空间直接加噪声
X_noisy, Y_noisy = add_noise_with_label(x_data, y_data, n=20, noise_range=0.1)

# 重新训练
params_noise, costdim = train(X_noisy, Y_noisy, n_h=8, lr=0.8)
acc_clean = test_accuracy(params_noise, x_data, y_data)
print(f"\n带噪声训练后，在原始样本上的准确率: {acc_clean:.1f}%")


#------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(np.arange(len(costdim))*500, costdim)
plt.title("training loss curve with noise")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.savefig("classification_cost_curve.png")
