import numpy as np
import matplotlib.pyplot as plt
import time, random, os

# Set Chinese font support
import matplotlib
# Set global font
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# Fix minus sign display
matplotlib.rcParams['axes.unicode_minus'] = False  # Correctly display minus signs

#------------------------------------------------------------
# 读取ASCII点阵数据文件，每行是一个字母，共26行
#------------------------------------------------------------
def load_ascii_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line: 
            continue
        row = [int(ch) for ch in line]
        data.append(row)
    return np.array(data, dtype=float)

#------------------------------------------------------------
# BP神经网络基本函数
#------------------------------------------------------------
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(int(time.time()))
    W1 = np.random.randn(n_h, n_x) * 0.5
    W2 = np.random.randn(n_y, n_h) * 0.5
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagate(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    Z1 = np.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2  # 线性输出层
    cache = {'A1': A1, 'A2': A2, 'Z1': Z1, 'Z2': Z2}
    return A2, cache

def calculate_cost(A2, Y):
    err = A2 - Y
    cost = np.mean(np.sum(err**2, axis=0))
    return cost

def backward_propagate(parameters, cache, X, Y):
    m = X.shape[0]
    W1, W2 = parameters['W1'], parameters['W2']
    A1, A2 = cache['A1'], cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(dZ1, X) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def update_parameters(parameters, grads, lr):
    for key in ['W1', 'b1', 'W2', 'b2']:
        parameters[key] -= lr * grads['d' + key]
    return parameters

#------------------------------------------------------------
# 训练函数
#------------------------------------------------------------
def train(X, Y, num_iterations=3000, learning_rate=0.1, hidden_dim=15, verbose=True):
    n_x, n_y = X.shape[1], X.shape[1]
    parameters = initialize_parameters(n_x, hidden_dim, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagate(X, parameters)
        cost = calculate_cost(A2, Y)
        grads = backward_propagate(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 200 == 0:
            costs.append(cost)
            if verbose:
                print(f"迭代 {i:4d} : cost = {cost:.6f}")

    return parameters, costs

#------------------------------------------------------------
# 主流程
#------------------------------------------------------------
def main():
    file_path = "ascii8x16.txt"
    if not os.path.exists(file_path):
        print("❌ ascii8x16.txt not found, please place it in the current directory.")
        return

    x_train = load_ascii_data(file_path)
    x_train = x_train / np.max(x_train)
    y_train = x_train.T

    #---------------------------------------------
    # 不同隐层节点个数的误差比较
    #---------------------------------------------
    hidden_nodes = [4, 8, 10, 15, 20, 30, 40]
    errors = []

    for hn in hidden_nodes:
        print(f"\n🔹 Start training with hidden nodes = {hn}")
        params, _ = train(x_train, y_train, num_iterations=2000, learning_rate=0.15, hidden_dim=hn, verbose=False)
        A2, _ = forward_propagate(x_train, params)
        final_err = calculate_cost(A2, y_train)
        errors.append(final_err)
        print(f"Hidden nodes {hn:2d} -> Final error {final_err:.6f}")

    plt.figure()
    plt.plot(hidden_nodes, errors, 'o-', linewidth=2)
    plt.title("Hidden Nodes vs Reconstruction Error")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.grid(True)
    plt.savefig("hidden_nodes_comparison.png")
    plt.close()

    #---------------------------------------------
    # 隐层=15时的重建图像展示
    #---------------------------------------------
    hn_show = 15
    params, _ = train(x_train, y_train, num_iterations=3000, learning_rate=0.15, hidden_dim=hn_show, verbose=True)
    A2, _ = forward_propagate(x_train, params)
    A2 = np.clip(A2.T, 0, 1)

    fig, axes = plt.subplots(5, 6, figsize=(10, 8))
    axes = axes.flatten()
    for i in range(min(26, len(axes))):
        img = A2[i].reshape(8, 16)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(chr(65+i))  # Display A~Z
        axes[i].axis('off')

    plt.suptitle(f"26 Letters Reconstruction with {hn_show} Hidden Nodes", fontsize=14)
    plt.tight_layout()
    plt.savefig("letter_reconstruction.png")
    plt.close()

#------------------------------------------------------------
# 程序入口
#------------------------------------------------------------
if __name__ == "__main__":
    main()
#============================================================
# END OF FILE
#============================================================
