import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math # 导入 math 库用于 sqrt

# ============================================================
# 第一部分：通用辅助函数
# ============================================================

def WTA2(x, w_flat):
    """ 
    Win-Take-All (赢者通吃)
    输入: x - 单个样本向量 (1x2)
          w_flat - 权重矩阵 (N x 2), N为神经元总数
    返回: id - 获胜神经元的索引
    """
    # 计算欧氏距离的平方：d^2 = sum((x - w)^2)
    diff = x - w_flat
    dist = np.sum(diff**2, axis=1)
    
    return np.argmin(dist)

# ============================================================
# 第二部分：1D 拓扑 (训练三角形数据)
# ============================================================

## 1D 数据生成
def generate_data_triangle(num):
    """
    在等边三角形区域内生成随机数据点。
    三角形底边在 x 轴上 [0, 1]，高为 sqrt(3)/2。
    """
    pointdim = []

    for _ in range(num):
        while True:
            # 限制采样区域为包含三角形的矩形 [0, 1] x [0, sqrt(3)/2]
            x = random.uniform(0, 1)
            y = random.uniform(0, math.sqrt(3) / 2)
            
            # 计算当前 x 坐标下，三角形边界的 y 值
            # 边界方程: y = sqrt(3) * (0.5 - abs(x-0.5))
            y_limit = math.sqrt(3) * (0.5 - abs(x - 0.5))
            
            if y > y_limit:
                continue # 如果点在三角形外部（上方），重新采样

            pointdim.append([x, y])
            break

    return np.array(pointdim)

## 1D 可视化
def show_data_1d(data, W, title=''):
    plt.clf()

    # 绘制训练数据
    plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='Train Data')
    
    # 绘制 SOFM 神经元 (权重)
    # 神经元点 (s=35, c='red')
    plt.scatter(W[:, 0], W[:, 1], s=35, c='red', label='SOFM Nodes')
    # 神经元连线 (拓扑结构)
    plt.plot(W[:, 0], W[:, 1], 'y-', linewidth=1, label='1D Topology')

    # 绘制三角形边界 (Cyan dashed lines)
    board_x = np.linspace(0, 1, 200)
    board_y = [math.sqrt(3) * (0.5 - abs(x - 0.5)) for x in board_x]
    plt.plot(board_x, board_y, 'c--', linewidth=1, label='Boundaries')
    plt.plot(board_x, np.zeros(len(board_x)), 'c--', linewidth=1) # 底部边界

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis([-0.05, 1.05, -0.05, .9])
    
    if len(title) > 0: 
        plt.title(title)
        
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

## 1D 邻域函数
def neighborid1(id, row, r):
    """
    获取 1D 线性拓扑下，获胜节点 id 的邻域索引
    In: id-获胜节点索引, row-总行数(神经元数), r-邻域半径
    """
    if r <= 0: return [id]

    iddim = []
    # 遍历范围 [id - r, id + r]
    for i in range(-r, r + 1):
        neighbor_id = id + i
        
        # 边界检查
        if 0 <= neighbor_id < row:
            iddim.append(neighbor_id)

    return iddim

## 1D 竞争更新
def compete1(x_dataset, w, eta, r):
    """
    1D 拓扑下的 SOFM 训练
    """
    for xx in x_dataset:
        # 1. 寻找获胜节点 (WTA)
        id = WTA2(xx, w) 

        # 2. 获取邻域索引
        iddim = neighborid1(id, w.shape[0], r)

        # 3. 更新获胜节点和邻域节点的权重
        for iidd in iddim:
            # 更新规则: w_new = w_old + eta * (x - w_old)
            w[iidd] = w[iidd] + eta * (xx - w[iidd])

    return w

# ============================================================
# 第三部分：2D 拓扑 (训练矩形数据)
# ============================================================

## 2D 数据生成
def generate_data_rect(num):
    """
    在单位矩形 [0, 1] x [0, 1] 内生成随机数据点
    """
    x_ = np.random.random(num)
    y_ = np.random.random(num)
    
    # 将 x 和 y 合并成 (num, 2) 的矩阵
    xy = np.vstack((x_, y_))
    return xy.T

## 2D 可视化
def show_data_2d(data):
    """
    绘制训练数据和边界
    """
    plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='Train Data')
    # 绘制矩形边界
    plt.vlines([0, 1], 0, 1, color='green', linewidth=1, linestyles='--')
    plt.hlines([0, 1], 0, 1, color='green', linewidth=1, linestyles='--')
    plt.axis([-0.1, 1.1, -0.1, 1.2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

def show_W_2d(w_grid, title=''):
    """
    绘制 2D SOFM 权重网格
    w_grid 的形状是 (ROWS, COLS, 2)
    """
    # 将网格展平为 (N*M, 2) 用于绘点
    w_flat = w_grid.reshape(-1, 2)
    plt.scatter(w_flat[:, 0], w_flat[:, 1], s=40, c='red', label='Node')

    # 绘制行连接 (沿着 COL 维度连接)
    for ww_row in w_grid:
        plt.plot(ww_row[:, 0], ww_row[:, 1], color='red', linewidth=1, linestyle='--')

    # 绘制列连接 (沿着 ROW 维度连接)
    for ww_col in w_grid.transpose((1, 0, 2)):
        plt.plot(ww_col[:, 0], ww_col[:, 1], color='red', linewidth=1, linestyle='--')

    plt.legend(loc="upper right")
    if len(title) > 0: 
        plt.title(title)
    plt.tight_layout()

## 2D WTA
def WTA2_2d(x, w_grid):
    """
    2D 拓扑下的 WTA，返回获胜节点的 (col_idx, row_idx)
    """
    ROWS, COLS, _ = w_grid.shape
    # 将网格展平为 (ROWS*COLS, 2)
    w_flat = w_grid.reshape(-1, 2)
    
    # 找到平面索引
    id_flat = WTA2(x, w_flat) 
    
    # 转换回 (col_idx, row_idx) 
    # 注意: 原代码的转换是 idy = id // col, idx = id % col，对应 (col, row)
    col_idx = id_flat % COLS
    row_idx = id_flat // COLS 
    
    return (col_idx, row_idx)

## 2D 邻域函数
def neighborid2(id_coord, row_num, col_num, r):
    """
    获取 2D 网格拓扑下，获胜节点 (col_idx, row_idx) 的 r 邻域索引
    In: id_coord-(col, row), row_num-总行数, col_num-总列数, r-邻域半径
    """
    if r <= 0: return [id_coord]

    iddim = []

    idx = id_coord[0] # 列索引
    idy = id_coord[1] # 行索引

    # 遍历行 (idy) 范围 [idy-r, idy+r]
    for i in range(-r, r + 1):
        if idy + i < 0 or idy + i >= row_num: 
            continue

        # 遍历列 (idx) 范围 [idx-r, idx+r]
        for j in range(-r, r + 1):
            if idx + j < 0 or idx + j >= col_num: 
                continue
                
            # 使用 (col_idx, row_idx) 格式
            iddim.append((idx + j, idy + i)) 

    return iddim

## 2D 竞争更新
def compete2(x_dataset, w_grid, eta, r):
    """
    2D 拓扑下的 SOFM 训练
    """
    ROWS, COLS, _ = w_grid.shape
    for xx in x_dataset:
        # 1. 寻找获胜节点 (WTA)，返回 (col, row)
        id_coord = WTA2_2d(xx, w_grid) 

        # 2. 获取邻域索引 (二维坐标列表)
        iddim = neighborid2(id_coord, ROWS, COLS, r)

        # 3. 更新获胜节点和邻域节点的权重
        for iidd in iddim:
            idx = iidd[0] # 列索引 (X)
            idy = iidd[1] # 行索引 (Y)
            
            # 更新规则: w_new = w_old + eta * (x - w_old)
            w_grid[idy, idx] = w_grid[idy, idx] + eta * (xx - w_grid[idy, idx])

    return w_grid


# ============================================================
# 第四部分：主程序运行
# ============================================================

def run_1d_som():
    print("--- 实验一：1D 拓扑 (训练三角形数据) ---")
    
    # --- 参数设置 ---
    SAMPLE_NUM = 300
    NEURAL_NUM = 100 # 100 个神经元排成 1D 链
    TRAIN_NUM = 200
    ETA_BEGIN = 0.3
    ETA_END = 0.01
    RATIO_BEGIN = 10 # 初始邻域半径
    RATIO_END = 0    # 最终邻域半径

    # --- 数据和权重初始化 ---
    random.seed(1)
    np.random.seed(1)
    x_data = generate_data_triangle(SAMPLE_NUM)
    
    # W 的形状是 (100, 2)
    W = np.random.rand(NEURAL_NUM, x_data.shape[1])
    W[:, 1] *= math.sqrt(3) / 2 # 限制初始 Y 坐标范围

    # --- 训练和可视化 ---
    plt.figure(figsize=(8, 6))

    for i in range(TRAIN_NUM):
        # 学习率 eta 线性衰减
        eta = (ETA_BEGIN - ETA_END) * (TRAIN_NUM - i) / (TRAIN_NUM - 1) + ETA_END
        # 邻域半径 ratio 线性衰减
        ratio = int((RATIO_BEGIN - RATIO_END) * (TRAIN_NUM - i) / (TRAIN_NUM - 1) + RATIO_END)

        # 打乱数据
        np.random.shuffle(x_data)
        
        # 竞争更新
        W = compete1(x_data, W, eta, ratio)

        if (i + 1) % 20 == 0 or i == TRAIN_NUM - 1:
            title = f"1D SOM - Step:{i+1}/{TRAIN_NUM}, R:{ratio}, eta:{eta:.4f}"
            show_data_1d(x_data, W, title)
            plt.show(block=False)
            plt.pause(0.01)

    plt.savefig("1d_som.png")
    print("1D 拓扑实验完成。")

# ------------------------------------------------------------

def run_2d_som():
    print("\n--- 实验二：2D 拓扑 (训练矩形数据) ---")
    
    # --- 参数设置 ---
    SAMPLE_NUM = 200
    ROWS = 5
    COLS = 5
    NEURAL_NUM = ROWS * COLS # 5x5 = 25 个神经元
    TRAIN_NUM = 100
    ETA_BEGIN = 0.1
    ETA_END = 0.01
    RATIO_BEGIN = 3 # 初始邻域半径
    RATIO_END = 0    # 最终邻域半径

    # --- 数据和权重初始化 ---
    random.seed(2)
    np.random.seed(2)
    x_data = generate_data_rect(SAMPLE_NUM)
    
    # W 的形状是 (ROWS, COLS, 2)
    # 随机初始化在 [0, 1] x [0, 1] 矩形内
    W_grid = np.random.rand(ROWS, COLS, 2) 

    # --- 训练和可视化 ---
    plt.figure(figsize=(8, 6))

    for i in range(TRAIN_NUM):
        # 学习率 eta 线性衰减
        eta = (ETA_BEGIN - ETA_END) * (TRAIN_NUM - i) / (TRAIN_NUM - 1) + ETA_END
        # 邻域半径 ratio 线性衰减
        ratio = int((RATIO_BEGIN - RATIO_END) * (TRAIN_NUM - i) / (TRAIN_NUM - 1) + RATIO_END)
        
        # 打乱数据
        np.random.shuffle(x_data)
        
        # 竞争更新
        W_grid = compete2(x_data, W_grid, eta, ratio)

        if (i + 1) % 10 == 0 or i == TRAIN_NUM - 1:
            plt.clf()
            show_data_2d(x_data)
            title = f"2D SOM - Step:{i+1}/{TRAIN_NUM}, R:{ratio}, eta:{eta:.4f}"
            show_W_2d(W_grid, title)
            plt.show(block=False)
            plt.pause(0.01)

    plt.savefig("2d_som.png")
    print("2D 拓扑实验完成。")


if __name__ == '__main__':
    # 运行 1D 拓扑实验 (训练三角形数据)
    run_1d_som()
    
    # 运行 2D 拓扑实验 (训练矩形数据)
    run_2d_som()