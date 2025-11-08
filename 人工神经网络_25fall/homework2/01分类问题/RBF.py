

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, dot, linalg, eye
from sklearn.cluster import KMeans

#------------------------------------------------------------
# 三类样本坐标（使用 PDF 第一组）
x_data = np.array([
    [0.75, 1.0], [0.5, 0.75], [0.25, 0.0],   # 类别1
    [0.5, 0.0],  [0.0, 0.0],  [1.0, 0.75],   # 类别2
    [1.0, 1.0],  [0.5, 0.25], [0.75, 0.5]    # 类别3
])
y_data = np.array([
    [1,-1,-1],[1,-1,-1],[1,-1,-1],
    [-1,1,-1],[-1,1,-1],[-1,1,-1],
    [-1,-1,1],[-1,-1,1],[-1,-1,1]
])

#------------------------------------------------------------
# 数据归一化到 [-1,1]
x_data = 2*x_data - 1

#------------------------------------------------------------
# 计算RBF隐层输出
# x: 输入样本
# H: 所有中心
# sigma: 高斯核宽度
def rbf_hide_out(x, H, sigma):
    Hx = H - x
    return np.array([exp(-dot(e,e)/(sigma**2)) for e in Hx])

#------------------------------------------------------------
# 正规化RBF网络
def regularized_rbf(x_data, y_data, sigma=0.5, lam=1e-3):
    # 每个样本点都是RBF中心
    centers = x_data.copy()
    # 构造隐层输出矩阵 H (row->center, col->sample)
    Hdim = np.array([rbf_hide_out(x, centers, sigma) for x in x_data]).T

    # 求解输出权矩阵 W = Y*(H'H + λI)^(-1)*H'
    W = dot(y_data.T, dot(linalg.inv(eye(Hdim.shape[0])*lam + dot(Hdim.T,Hdim)), Hdim.T))

    # 预测
    yy = dot(W, Hdim)
    yy1 = np.array([[1 if e>0 else -1 for e in l] for l in yy])
    err = [1 if any(x1!=x2) else 0 for x1,x2 in zip(yy1.T, y_data)]

    print("=== 正规化 RBF 分类结果 ===")
    print("预测输出：\n", yy1)
    print("错误样本数:", sum(err))
    print("-----------------------------")
    return W, centers, Hdim, yy1

#------------------------------------------------------------
# 广义RBF网络
def generalized_rbf(x_data, y_data, n_hidden=3, sigma=0.5, lam=1e-3):
    # 用 KMeans 聚类选择 RBF 中心
    km = KMeans(n_clusters=n_hidden, random_state=0).fit(x_data)
    centers = km.cluster_centers_

    # 构造隐层输出矩阵 H (row->center, col->sample)
    Hdim = np.array([rbf_hide_out(x, centers, sigma) for x in x_data]).T

    # 求输出权
    # 修正矩阵形状匹配问题
    Hdim_T_Hdim = dot(Hdim.T, Hdim)
    regularization_matrix = eye(Hdim_T_Hdim.shape[0]) * lam
    W = dot(y_data.T, dot(linalg.inv(regularization_matrix + Hdim_T_Hdim), Hdim.T))

    # 预测
    yy = dot(W, Hdim)
    yy1 = np.array([[1 if e>0 else -1 for e in l] for l in yy])
    err = [1 if any(x1!=x2) else 0 for x1,x2 in zip(yy1.T, y_data)]

    print(f"=== 广义 RBF 分类结果（隐层节点 = {n_hidden}） ===")
    print("聚类中心:\n", centers)
    print("预测输出:\n", yy1)
    print("错误样本数:", sum(err))
    print("-----------------------------")

    return W, centers, Hdim, yy1

#------------------------------------------------------------
# 仿真执行

# 1️⃣ 正规化 RBF 网络
W_reg, centers_reg, H_reg, y_pred_reg = regularized_rbf(x_data, y_data, sigma=0.5)

# 2️⃣ 广义 RBF 网络：隐层节点 2, 3, 4
for n_hidden in [2, 3, 4]:
    generalized_rbf(x_data, y_data, n_hidden=n_hidden, sigma=0.5)

#------------------------------------------------------------
# 可选：可视化分类结果
plt.figure()
plt.scatter(x_data[:3,0], x_data[:3,1], c='r', marker='o', label='Class 1')
plt.scatter(x_data[3:6,0], x_data[3:6,1], c='g', marker='^', label='Class 2')
plt.scatter(x_data[6:,0], x_data[6:,1], c='b', marker='s', label='Class 3')
plt.title("3-Class Samples for RBF Classification")
plt.legend()
plt.grid(True)
plt.savefig("RBF_classification.png")

#============================================================
# END OF FILE
#============================================================
