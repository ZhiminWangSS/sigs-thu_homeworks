import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# 通用数学与辅助函数
# ============================================================

def f_hermit(x):
    """目标函数: Hermit Function"""
    return 1.1 * (1 - x + 2 * x**2) * np.exp(-x**2/2)

def WTA_nearest(x, v):
    """
    Win-Take-All: 找到与输入 x 最近的神经元 v 的索引
    支持 1D (标量距离) 和 2D (欧氏距离)
    """
    # 如果 v 是 1D 数组 (单向 CPN)
    if v.ndim == 1:
        dist = (x - v)**2
    # 如果 v 是 2D 数组 (双向 CPN, 每一行是一个向量)
    else:
        # 计算 x (向量) 与 v 中每个向量的距离
        diff = x - v
        dist = np.sum(diff**2, axis=1)
    
    return np.argmin(dist)

def K_neighbor(v, x_dataset, eta):
    """
    Kohonen Layer 竞争学习 (Clustering)
    v: 神经元权重 (聚类中心)
    x_dataset: 训练数据
    eta: 学习率
    """
    # 复制一份 v 防止修改原引用（虽然此处需要修改，但保持副本习惯更安全，这里直接改原引用也行）
    # 这里的逻辑是遍历每个样本，更新最近的节点
    for xx in x_dataset:
        id = WTA_nearest(xx, v)
        v[id] = v[id] + eta * (xx - v[id])
    return v

def CPN_v_out(x_dataset, v):
    """
    计算隐层输出 H (One-hot 向量)
    对于每个输入样本，最近的神经元输出 1，其余为 0
    """
    H = []
    for xx in x_dataset:
        h = np.zeros(v.shape[0])
        id = WTA_nearest(xx, v)
        h[id] = 1
        H.append(h)
    return np.array(H).T  # 转置，形状变为 (Hidden_Nodes, Samples)

def CPN_W(h, y):
    """
    Grossberg Layer 权重计算 (使用伪逆/最小二乘法)
    W = Y * H.T * inv(H * H.T)
    """
    # 添加微小正则项防止矩阵奇异
    vv = h.dot(h.T) + 0.000001 * np.eye(h.shape[0]) 
    # 计算伪逆部分
    vvv = np.linalg.inv(vv).dot(h)
    # 计算权重 W
    y_reshaped = y.reshape(1, -1) # 确保 y 是行向量
    return y_reshaped.dot(vvv.T)

def plot_base(ax, x_train, y_train):
    """绘制基础背景：目标函数曲线和训练数据点"""
    x_line = np.linspace(-4, 4, 250)
    ax.clear()
    ax.plot(x_line, f_hermit(x_line), '--', c='grey', linewidth=1, label='Hermit Func')
    ax.scatter(x_train, y_train, s=10, c='darkviolet', alpha=0.5, label='Train Data')
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)

# ============================================================
# 实验 1: 单向 CPN (Unidirectional)
# ============================================================

def run_unidirectional():
    print("--- 开始运行：单向 CPN (Unidirectional) ---")
    
    # 1. 数据准备
    TRAIN_DATA_NUM = 500
    np.random.seed(1) # 固定种子以便复现
    x_train = np.random.uniform(-4, 4, TRAIN_DATA_NUM)
    y_train = f_hermit(x_train)
    
    # 2. 网络初始化
    NODE_NUM = 50
    # 初始化竞争层权重：直接从训练数据中选取前50个点作为初始中心
    v_data = np.copy(x_train[0:NODE_NUM]) 
    
    # 3. 训练参数
    ETA_BEGIN = 0.1
    ETA_END = 0.0
    TRAIN_STEP = 50 # 适当减少步数以加快演示
    
    # 4. 绘图设置
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title('Experiment 1: Unidirectional CPN')

    # 5. 训练循环
    etas = np.linspace(ETA_BEGIN, ETA_END, TRAIN_STEP)
    
    for i, eta in enumerate(etas):
        # --- A. 竞争层训练 (Unsupervised Clustering on X) ---
        v_data = K_neighbor(v_data, x_train, eta)
        
        # --- B. 映射层计算 (Supervised Weight Calculation) ---
        # 计算隐层对所有训练数据的响应 H
        H = CPN_v_out(x_train, v_data)
        # 计算输出层权重 W (一次性计算最优解)
        W = CPN_W(H, y_train)
        
        # --- C. 绘图与验证 ---
        if i % 5 == 0 or i == TRAIN_STEP - 1: # 每5步刷新一次绘图
            plot_base(ax, x_train, y_train)
            
            # 绘制网络拟合曲线
            x_line = np.linspace(-4, 4, 500)
            # 对于单向，直接用 x 聚类中心进行预测
            Hx = CPN_v_out(x_line, v_data) 
            y_line = W.dot(Hx)
            
            ax.plot(x_line, y_line[0], c='red', linewidth=2, label='Net Performance')
            
            # 绘制聚类中心 (在单向中，这些点位于 x 轴上，我们把它们映射到曲线上显示位置)
            v_H = CPN_v_out(v_data, v_data)
            v_yy = W.dot(v_H)
            ax.scatter(v_data, v_yy[0], s=30, c='darkcyan', zorder=5, label='Nodes (X-Clustered)')

            ax.set_title(f'Unidirectional Step:{i}, Eta:{eta:.3f}')
            ax.legend(loc='upper right')
            plt.pause(0.01)

    print("单向 CPN 训练完成。")
    plt.savefig("Unidirectional.png")

# ============================================================
# 实验 2: 双向 CPN (Bidirectional)
# ============================================================

def run_bidirectional():
    print("\n--- 开始运行：双向 CPN (Bidirectional) ---")
    
    # 1. 数据准备
    TRAIN_DATA_NUM = 500
    np.random.seed(2)
    x_train = np.random.uniform(-4, 4, TRAIN_DATA_NUM)
    y_train = f_hermit(x_train)
    
    # 构建联合空间数据 (X, Y)
    xy_train = np.column_stack((x_train, y_train))
    
    # 2. 网络初始化
    NODE_NUM = 50
    # 初始化竞争层权重：从训练数据联合空间中选取
    v_seed_x = x_train[0:NODE_NUM]
    v_seed_y = f_hermit(v_seed_x)
    v_data = np.column_stack((v_seed_x, v_seed_y)) # 形状 (50, 2)
    
    # 3. 训练参数
    ETA_BEGIN = 0.1
    ETA_END = 0.0
    TRAIN_STEP = 50
    
    # 4. 绘图设置
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title('Experiment 2: Bidirectional CPN')

    # 5. 训练循环
    etas = np.linspace(ETA_BEGIN, ETA_END, TRAIN_STEP)
    
    for i, eta in enumerate(etas):
        # --- A. 竞争层训练 (Unsupervised Clustering on X and Y) ---
        # 注意：这里传入的是 xy_train (2D数据)
        v_data = K_neighbor(v_data, xy_train, eta)
        
        # --- B. 映射层计算 ---
        # 关键点：虽然聚类利用了 Y 信息，但函数逼近预测时只能利用 X 信息。
        # 因此，我们建立映射时，隐层的激活仅基于 v_data 的 X 分量与 输入 X 的距离。
        v_data_x = v_data[:, 0] # 取出聚类中心的 X 坐标
        
        H = CPN_v_out(x_train, v_data_x)
        W = CPN_W(H, y_train)
        
        # --- C. 绘图与验证 ---
        if i % 5 == 0 or i == TRAIN_STEP - 1:
            plot_base(ax, x_train, y_train)
            
            # 绘制网络拟合曲线
            x_line = np.linspace(-4, 4, 500)
            Hx = CPN_v_out(x_line, v_data_x)
            y_line = W.dot(Hx)
            
            ax.plot(x_line, y_line[0], c='red', linewidth=2, label='Net Performance')
            
            # 绘制隐层节点 (Hide Node)
            # 双向 CPN 中，节点实际上是在 (x,y) 平面上移动的
            ax.scatter(v_data[:, 0], v_data[:, 1], s=30, c='green', zorder=5, label='Nodes (XY-Clustered)')
            
            ax.set_title(f'Bidirectional Step:{i}, Eta:{eta:.3f}')
            ax.legend(loc='upper right')
            plt.pause(0.01)

    print("双向 CPN 训练完成。")
    plt.savefig("Bidirectional.png")

# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 依次运行两个实验
    # 关闭第一个窗口后，会自动运行第二个
    run_unidirectional()
    run_bidirectional()