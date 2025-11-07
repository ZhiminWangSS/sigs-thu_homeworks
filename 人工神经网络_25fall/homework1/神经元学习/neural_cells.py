import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置字体（Windows/Mac/Linux通用）
rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题


# === 1. 数据准备 ===
xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
ddim = [-1, 1, -1, -1, 1, 1]
eta = 0.5
epochs = 2

# 激活函数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t * t

def sign(x):
    return np.where(x >= 0, 1, -1)

# === 2. 统一神经元学习类 ===
class UnifiedNeuron:
    def __init__(self, algorithm='hebbian', learning_rate=0.5, init_weights=None):
        self.eta = learning_rate
        self.algorithm = algorithm.lower()
        if init_weights is None:
            self.w = np.array([0.0, 0.0, 0.0])  # [b, w1, w2]
        else:
            self.w = np.array(init_weights, dtype=float)
    
    def activate(self, net):
        if self.algorithm == 'perceptron':
            return sign(net)
        else:
            return tanh(net)
    
    def update(self, x, d):
        x_vec = np.array([1.0, x[0], x[1]])  # [bias, x1, x2]
        net = np.dot(self.w, x_vec)
        
        if self.algorithm == 'hebbian':
            o = tanh(net)
            r = o
        elif self.algorithm == 'perceptron':
            o = sign(net)
            r = d - o
        elif self.algorithm == 'delta':
            o = tanh(net)
            deriv = tanh_derivative(net)
            r = (d - o) * deriv
        elif self.algorithm == 'widrow-hoff':
            # LMS: linear output, no activation
            o = net
            r = d - o
        elif self.algorithm == 'correlation':
            r = d
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        delta_w = self.eta * r * x_vec
        self.w += delta_w
    
    def train(self, x_data, d_data, epochs=2):
        for _ in range(epochs):
            for x, d in zip(x_data, d_data):
                self.update(x, d)
        return self.w.copy()

# === 3. 执行五种算法 ===
algorithms = ['hebbian', 'perceptron', 'delta', 'widrow-hoff', 'correlation']
results = {}

for alg in algorithms:
    neuron = UnifiedNeuron(algorithm=alg, learning_rate=eta)
    w_final = neuron.train(xdim, ddim, epochs=epochs)
    results[alg] = w_final  # [b, w1, w2]

# 打印结果
print("经过 2 轮训练后的权重结果 (w1, w2, b)：")
for alg, w in results.items():
    b, w1, w2 = w
    print(f"{alg.capitalize():15s}: w1 = {w1:8.4f}, w2 = {w2:8.4f}, b = {b:8.4f}")

# === 4. 可视化：绘制 (w1, w2) 权重向量 ===
plt.figure(figsize=(8, 6))

# # 绘制样本点
# for (x1, x2), d in zip(xdim, ddim):
#     if d == 1:
#         plt.scatter(x1, x2, c='blue', marker='o', s=100, label='Class +1' if 'Class +1' not in plt.gca().get_legend_handles_labels()[1] else "")
#     else:
#         plt.scatter(x1, x2, c='red', marker='+', s=100, linewidths=2, label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

# 绘制各算法的 (w1, w2) 向量（从原点出发）
colors = ['green', 'purple', 'orange', 'brown', 'pink']
for i, (alg, w) in enumerate(results.items()):
    b, w1, w2 = w
    plt.arrow(0, 0, w1, w2, 
              head_width=0.02, head_length=0.03, 
              fc=colors[i], ec=colors[i], 
              linewidth=2, 
              label=f'{alg.capitalize()} (w1,w2)')

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlim(-0.8, 1.6)
plt.ylim(-0.25, 1.0)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('训练样本与各算法学习到的权重向量 (w1, w2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('neural_cells_weights.png')
plt.show()