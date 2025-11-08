#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人工神经网络第二次作业 02-回归问题（Peaks函数逼近）
实现：BP 网络 + RBF 网络
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# Peaks函数定义（与MATLAB一致）
def peaks(x, y):
    return (3 * (1 - x)**2 * np.exp(-(x**2 + (y + 1)**2))
            - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2 + y**2))
            - 1/3 * np.exp(-((x + 1)**2 + y**2)))

# ------------------------------------------------------------
# ========== BP 网络部分 ==========
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(0)
    W1 = np.random.randn(n_h, n_x) * 0.5
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.5
    b2 = np.zeros((n_y, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def forward_propagate(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    Z1 = np.dot(W1, X.T) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    cache = {'A1': A1, 'A2': A2}
    return A2, cache

def calculate_cost(A2, Y):
    return np.mean((A2 - Y)**2)

def backward_propagate(parameters, cache, X, Y):
    m = X.shape[0]
    W2 = parameters['W2']
    A1, A2 = cache['A1'], cache['A2']
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1**2)
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(parameters, grads, lr):
    for key in ['W1', 'b1', 'W2', 'b2']:
        parameters[key] -= lr * grads['d' + key]
    return parameters

def train_bp(X, Y, num_iterations=4000, n_h=7, lr=0.35, print_cost=True):
    n_x, n_y = 2, 1
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(num_iterations):
        A2, cache = forward_propagate(X, parameters)
        cost = calculate_cost(A2, Y)
        grads = backward_propagate(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, lr)
        if print_cost and i % 200 == 0:
            print(f"[BP] 迭代 {i} 次后误差: {cost:.4f}")
            costs.append(cost)
    return parameters, costs

# ------------------------------------------------------------
# ========== RBF 网络部分 ==========
def gaussian_rbf(x, c, sigma):
    return np.exp(-np.sum((x - c)**2, axis=1) / (2 * sigma**2))

def build_rbf_design_matrix(X, centers, sigma):
    G = np.zeros((X.shape[0], centers.shape[0]))
    for i, c in enumerate(centers):
        G[:, i] = gaussian_rbf(X, c, sigma)
    return G

def train_rbf(X, Y, n_centers=20, sigma=1.0):
    # 随机选取中心
    idx = np.random.choice(X.shape[0], n_centers, replace=False)
    centers = X[idx, :]

    # 构建高斯基函数矩阵
    G = build_rbf_design_matrix(X, centers, sigma)

    # 使用最小二乘法求解权值
    W = np.linalg.pinv(G).dot(Y.T)
    return centers, W, sigma

def predict_rbf(X, centers, W, sigma):
    G = build_rbf_design_matrix(X, centers, sigma)
    Y_pred = G.dot(W)
    return Y_pred.T

# ------------------------------------------------------------
# ========== 主程序 ==========
if __name__ == '__main__':
    # 生成训练样本
    SAMPLE_NUM = 300
    xs = np.random.uniform(-4, 4, SAMPLE_NUM)
    ys = np.random.uniform(-4, 4, SAMPLE_NUM)
    zs = peaks(xs, ys)
    X_train = np.vstack((xs, ys)).T
    Y_train = zs.reshape(1, -1)

    # 训练 BP 网络
    bp_params, bp_costs = train_bp(X_train, Y_train, num_iterations=4000, n_h=7, lr=0.35)

    # 绘制BP误差曲线
    plt.figure()
    plt.plot(np.arange(len(bp_costs))*200, bp_costs, 'r-', label='BP误差')
    plt.xlabel("训练步数")
    plt.ylabel("均方误差")
    plt.title("BP网络训练误差曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 训练 RBF 网络
    centers, W, sigma = train_rbf(X_train, Y_train, n_centers=30, sigma=1.2)
    Y_pred_rbf = predict_rbf(X_train, centers, W, sigma)
    mse_rbf = np.mean((Y_pred_rbf - Y_train)**2)
    print(f"[RBF] 训练样本拟合误差 MSE = {mse_rbf:.4f}")

    # ========== 绘制拟合曲面 ==========
    X = np.arange(-4, 4, 0.1)
    Y = np.arange(-4, 4, 0.1)
    XX, YY = np.meshgrid(X, Y)
    XY = np.array([XX.ravel(), YY.ravel()]).T

    # BP 网络预测
    Z_bp, _ = forward_propagate(XY, bp_params)
    ZZ_bp = Z_bp.reshape(XX.shape)

    # RBF 网络预测
    Z_rbf = predict_rbf(XY, centers, W, sigma)
    ZZ_rbf = Z_rbf.reshape(XX.shape)

    # 绘制 BP 拟合图
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(XX, YY, ZZ_bp, cmap='coolwarm', linewidth=0, antialiased=False)
    ax1.scatter(xs, ys, zs, color='r', s=10)
    ax1.set_title("BP network regression")

    # 绘制 RBF 拟合图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(XX, YY, ZZ_rbf, cmap='viridis', linewidth=0, antialiased=False)
    ax2.scatter(xs, ys, zs, color='r', s=10)
    ax2.set_title(f"RBF network regression (σ={sigma})")

    plt.tight_layout()
    plt.savefig("peak_regression.png")

    # ========== 对比实验 ==========
    
    # BP网络：不同训练样本数量对比
    sample_sizes = [50, 150, 300]
    fig_bp_samples = plt.figure(figsize=(15, 6))
    
    for i, sample_size in enumerate(sample_sizes):
        # 生成不同数量的训练样本
        xs_bp = np.random.uniform(-4, 4, sample_size)
        ys_bp = np.random.uniform(-4, 4, sample_size)
        zs_bp = peaks(xs_bp, ys_bp)
        X_train_bp = np.vstack((xs_bp, ys_bp)).T
        Y_train_bp = zs_bp.reshape(1, -1)
        
        # 训练BP网络
        bp_params_sample, _ = train_bp(X_train_bp, Y_train_bp, num_iterations=2000, n_h=7, lr=0.35, print_cost=False)
        
        # 预测
        Z_bp_sample, _ = forward_propagate(XY, bp_params_sample)
        ZZ_bp_sample = Z_bp_sample.reshape(XX.shape)
        
        # 绘制
        ax = fig_bp_samples.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(XX, YY, ZZ_bp_sample, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=False)
        ax.scatter(xs_bp, ys_bp, zs_bp, color='r', s=10, alpha=0.6)
        ax.set_title(f"BP: {sample_size} samples")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig("bp_samples_comparison.png")
 
    # RBF网络：不同尺度参数对比
    sigma_values = [0.5, 1.2, 2.0]
    fig_rbf_sigma = plt.figure(figsize=(15, 6))
    
    for i, sigma_val in enumerate(sigma_values):
        # 训练RBF网络
        centers_sigma, W_sigma, _ = train_rbf(X_train, Y_train, n_centers=30, sigma=sigma_val)
        
        # 预测
        Z_rbf_sigma = predict_rbf(XY, centers_sigma, W_sigma, sigma_val)
        ZZ_rbf_sigma = Z_rbf_sigma.reshape(XX.shape)
        
        # 计算训练误差
        Y_pred_rbf_sigma = predict_rbf(X_train, centers_sigma, W_sigma, sigma_val)
        mse_rbf_sigma = np.mean((Y_pred_rbf_sigma - Y_train)**2)
        
        # 绘制
        ax = fig_rbf_sigma.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(XX, YY, ZZ_rbf_sigma, cmap='viridis', alpha=0.8, linewidth=0, antialiased=False)
        ax.scatter(xs, ys, zs, color='r', s=10, alpha=0.6)
        ax.set_title(f"RBF: σ={sigma_val}\nMSE={mse_rbf_sigma:.4f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig("rbf_sigma_comparison.png")
    
    # RBF网络：不同隐层神经元数量对比
    center_counts = [10, 30, 50]
    fig_rbf_centers = plt.figure(figsize=(15, 6))
    
    for i, n_centers in enumerate(center_counts):
        # 训练RBF网络
        centers_count, W_count, sigma_count = train_rbf(X_train, Y_train, n_centers=n_centers, sigma=1.2)
        
        # 预测
        Z_rbf_count = predict_rbf(XY, centers_count, W_count, sigma_count)
        ZZ_rbf_count = Z_rbf_count.reshape(XX.shape)
        
        # 计算训练误差
        Y_pred_rbf_count = predict_rbf(X_train, centers_count, W_count, sigma_count)
        mse_rbf_count = np.mean((Y_pred_rbf_count - Y_train)**2)
        
        # 绘制
        ax = fig_rbf_centers.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(XX, YY, ZZ_rbf_count, cmap='plasma', alpha=0.8, linewidth=0, antialiased=False)
        ax.scatter(xs, ys, zs, color='r', s=10, alpha=0.6)
        ax.set_title(f"RBF: {n_centers} centers\nMSE={mse_rbf_count:.4f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig("rbf_centers_comparison.png")
    
    print("对比实验完成！生成了以下图表：")
    print("- bp_samples_comparison.png: BP网络不同样本数量对比")
    print("- rbf_sigma_comparison.png: RBF网络不同尺度参数对比")
    print("- rbf_centers_comparison.png: RBF网络不同隐层神经元数量对比")
