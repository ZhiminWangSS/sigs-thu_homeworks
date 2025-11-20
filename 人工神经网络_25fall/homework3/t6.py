import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# 1. 数据准备 (Data Preparation)
# ============================================================

def normalize_data(data):
    """将数据归一化到 [0, 1] 范围，方便 SOM 训练"""
    data = np.array(data, dtype=float)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    # 防止除以0
    diff = max_val - min_val
    diff[diff == 0] = 1
    return (data - min_val) / diff

# --- 数据集 1: 参考代码中的原始数据 ---
raw_data_1 = [
    [246,53], [408,79], [909,89], [115,264], [396,335], 
    [185,456], [699,252], [963,317], [922,389], [649,515]
]

# ============================================================
# 1. 数据准备 (Data Preparation)
# ============================================================
# 第一种位置分布
raw_data_2 = [
    [349, 198], [268, 510], [736, 381], [1048, 187], [924, 480],
    [969, 682], [1034, 793], [597, 754], [631, 556], [173, 304]
]

# 第二种位置分布
raw_data_3 = [
    [297, 338], [403, 604], [736, 381], [1039, 286], [668, 553],
    [929, 598], [900, 137], [606, 761], [304, 448], [521, 430]
]

# 第三种位置分布
raw_data_4 = [
    [369, 170], [713, 415], [742, 600], [828, 325], [876, 675],
    [106, 340], [1038, 803], [845, 823], [1165, 151], [546, 814]
]


# ============================================================
# 2. 核心算法函数 (SOM Algorithm)
# ============================================================

def get_winner(x, w):
    """
    Win-Take-All: 找到与输入样本 x 距离最近的神经元索引
    """
    # 计算欧氏距离平方
    diff = x - w
    dist = np.sum(diff**2, axis=1)
    return np.argmin(dist)

def get_ring_neighbors(winner_id, total_neurons, radius):
    """
    获取环形拓扑(Ring Topology)下的邻域索引
    核心逻辑：处理首尾相连 (Wrap-around)
    """
    if radius <= 0:
        return [winner_id]
    
    indices = []
    for i in range(-radius, radius + 1):
        # 使用模运算处理环形结构 (相当于 reference 代码中的 if < 0 += row)
        neighbor_idx = (winner_id + i) % total_neurons
        indices.append(neighbor_idx)
        
    return indices

def update_weights(x, w, neighbor_indices, eta):
    """
    更新权重
    """
    for idx in neighbor_indices:
        w[idx] = w[idx] + eta * (x - w[idx])
    return w

def plot_tsp(ax, city_data, neuron_weights, step, total_steps, radius, eta):
    """
    绘图函数
    """
    ax.clear()
    
    # 1. 画城市 (样本点)
    ax.scatter(city_data[:, 0], city_data[:, 1], s=30, c='blue', label='Cities (View Site)')
    
    # 2. 画神经元 (路径)
    # 为了显示闭环，将第一个点追加到末尾
    plot_weights = np.vstack((neuron_weights, neuron_weights[0]))
    
    ax.plot(plot_weights[:, 0], plot_weights[:, 1], 'r-', linewidth=1.5, alpha=0.6, label='Path')
    ax.scatter(neuron_weights[:, 0], neuron_weights[:, 1], s=20, c='red', marker='x', label='Neurons')
    
    ax.set_title(f"Step: {step}/{total_steps} | R: {radius} | Eta: {eta:.3f}")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

# ============================================================
# 3. 主运行逻辑 (Main Loop)
# ============================================================

def run_tsp_som(raw_data, dataset_name="Dataset"):
    # 1. 数据预处理
    x_data = normalize_data(raw_data)
    
    # 2. 参数设置
    SAMPLE_NUM = x_data.shape[0]
    # 在 TSP SOM 中，神经元数量通常等于或略大于城市数量
    NEURAL_NUM = SAMPLE_NUM 
    
    # 初始化权重 (神经元位置)，随机分布在 [0,1] 空间
    np.random.seed(1) # 固定随机种子以复现结果
    W = np.random.rand(NEURAL_NUM, x_data.shape[1])

    # 训练参数
    TRAIN_NUM = 500        # 迭代次数，可以根据数据量适当增加
    ETA_BEGIN = 0.5        # 初始学习率
    ETA_END = 0.01         # 结束学习率
    RATIO_BEGIN = int(NEURAL_NUM * 0.2) # 初始邻域半径 (通常设为神经元总数的 10%-20%)
    RATIO_END = 0          # 结束邻域半径

    # 3. 设置绘图
    plt.ion() # 开启交互模式
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title(f"TSP Solution - {dataset_name}")

    print(f"--- Start Training: {dataset_name} ---")
    
    # 4. 训练循环
    for i in range(TRAIN_NUM):
        # 线性衰减参数
        progress = i / (TRAIN_NUM - 1)
        eta = ETA_BEGIN - (ETA_BEGIN - ETA_END) * progress
        ratio = int(RATIO_BEGIN - (RATIO_BEGIN - RATIO_END) * progress)
        
        # 随机打乱数据顺序 (防止输入顺序影响)
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        x_data_shuffled = x_data[indices]
        
        # 遍历样本进行竞争学习
        for xx in x_data_shuffled:
            # (1) 寻找获胜节点
            winner_id = get_winner(xx, W)
            
            # (2) 获取环形邻域
            neighbors = get_ring_neighbors(winner_id, NEURAL_NUM, ratio)
            
            # (3) 更新权重
            W = update_weights(xx, W, neighbors, eta)
        
        # 动态绘图 (每隔一定步数刷新，加快运行速度)
        if i % 20 == 0 or i == TRAIN_NUM - 1:
            plot_tsp(ax, x_data, W, i+1, TRAIN_NUM, ratio, eta)
            plt.pause(0.01)

    plt.ioff()
    print(f"--- Finished: {dataset_name} ---\n")
    plt.savefig(f"{dataset_name}.png")

# ============================================================
# 4. 执行入口
# ============================================================

if __name__ == "__main__":
    # 请根据作业要求，依次解除注释运行
    
    # --- 运行示例数据 ---
    run_tsp_som(raw_data_1, "Dataset 1 (Example)")
    
    # --- 运行 PDF 测试数据 1 ---
    run_tsp_som(raw_data_2, "Dataset 2 (Circle)")
    
    # --- 运行 PDF 测试数据 2 ---
    run_tsp_som(raw_data_3, "Dataset 3 (Random)")
    
    run_tsp_som(raw_data_4, "Dataset 4 (Random)")