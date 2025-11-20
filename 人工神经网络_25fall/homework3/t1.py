import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. 数据准备
# -------------------------
def angle_to_vec(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)], dtype=np.float32)

# 初始权向量
W = np.array([
    angle_to_vec(45),
    angle_to_vec(155),
    angle_to_vec(300)
], dtype=np.float32)

# 八个训练样本
sample_angles = [185, 175, 160, 270, 250, 240, 30, 60]
X = np.array([angle_to_vec(a) for a in sample_angles], dtype=np.float32)


# -------------------------
# 2. 归一化
# -------------------------
def normalize(v):
    return v / np.linalg.norm(v)


# -------------------------
# 3. WTA 赢家选择
# -------------------------
def winner(x, W):
    inner = W @ x
    return np.argmax(inner)


# -------------------------
# 4. 单次（epoch）竞争学习
# -------------------------
def train_once(W, X, eta):
    W_new = W.copy()
    for x in X:
        k = winner(x, W_new)
        W_new[k] = normalize(W_new[k] + eta * (x - W_new[k]))
    return W_new


# -------------------------
# 5. 两种学习率策略
# -------------------------
def train_loop(W_init, X, epochs, mode="geometric"):
    W = W_init.copy()
    history = []
    print(f"初始权重:\n", W)
    eta0 = 0.6
    if mode == "geometric":
        # 几何下降 η_{n+1} = 0.75 * η_n
        eta = eta0
        for i in range(epochs):
            W = train_once(W, X, eta)
            history.append(W.copy())
            if i == 0:
                print(f"第一次迭代后权重 (几何下降):\n", W)
            eta = eta * 0.75
        return W, history

    elif mode == "linear":
        # 线性下降： η = 0.8 * (N-n)/N
        N = epochs
        for n in range(N):
            eta = 0.8 * (N - n) / N
            W = train_once(W, X, eta)
            history.append(W.copy())
            if n == 0:
                print(f"第一次迭代后权重 (线性下降):\n", W)
        return W, history


# -------------------------
# 6. 运行训练
# -------------------------
W0 = W.copy()

W_geo, hist_geo = train_loop(W0, X, 100, mode="geometric")
W_lin, hist_lin = train_loop(W0, X, 100, mode="linear")

print("最终权向量 (几何下降):\n", W_geo)
print("最终权向量 (线性下降):\n", W_lin)

# -------------------------
# 7. 绘制结果
# -------------------------
def plot_results(X, W_init, W_geo, W_lin, sample_angles):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制训练数据点
    ax1.scatter(X[:, 0], X[:, 1], c='blue', label='data', s=80, alpha=0.7)
    # 绘制初始权重向量
    ax1.quiver([0, 0, 0], [0, 0, 0], W_init[:, 0], W_init[:, 1], 
               color='red', scale=1, scale_units='xy', angles='xy', 
               label='init weight', width=0.01)
    # 绘制几何下降收敛权重向量
    ax1.quiver([0, 0, 0], [0, 0, 0], W_geo[:, 0], W_geo[:, 1], 
               color='green', scale=1, scale_units='xy', angles='xy', 
               label='equivalence weight', width=0.01)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('equivalence drop')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # 绘制训练数据点
    ax2.scatter(X[:, 0], X[:, 1], c='blue', label='data', s=80, alpha=0.7)
    # 绘制初始权重向量
    ax2.quiver([0, 0, 0], [0, 0, 0], W_init[:, 0], W_init[:, 1], 
               color='red', scale=1, scale_units='xy', angles='xy', 
               label='init weight', width=0.01)
    # 绘制线性下降收敛权重向量
    ax2.quiver([0, 0, 0], [0, 0, 0], W_lin[:, 0], W_lin[:, 1], 
               color='purple', scale=1, scale_units='xy', angles='xy', 
               label='linear weight', width=0.01)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('linear')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig("./f1.png")

# 绘制结果
plot_results(X, W0, W_geo, W_lin, sample_angles)
