import numpy as np
import random
import matplotlib.pyplot as plt


class MultiClassPerceptron:
    def __init__(self, n_classes=7, n_features=63, eta=0.5):
        self.n_classes = n_classes
        self.eta = eta
        # 每个类一个权重向量 (b, w1, ..., w63)
        self.weights = np.zeros((n_classes, n_features + 1))  # shape: (7, 64)

    def predict(self, x):
        """x: (63,) → 返回预测类别索引"""
        x_bias = np.concatenate([[1], x])  # (64,)
        outputs = np.dot(self.weights, x_bias)  # (7,)
        return np.argmax(outputs)  # 返回最大输出的索引

    def train(self, X, y, epochs=10):
        """X: (n_samples, 63), y: (n_samples, 7) one-hot"""
        for epoch in range(epochs):
            errors = 0
            for i in range(len(X)):
                x = X[i]
                true_label = np.argmax(y[i])  # 0~6
                x_bias = np.concatenate([[1], x])
                outputs = np.dot(self.weights, x_bias)
                pred = np.argmax(outputs)

                # 对每个神经元更新
                for c in range(self.n_classes):
                    d = 1 if c == true_label else -1
                    o = 1 if outputs[c] >= 0 else -1
                    if o != d:
                        self.weights[c] += self.eta * (d - o) * x_bias
                        errors += 1
            if errors == 0:
                print(f"✅ 第 {epoch+1} 轮收敛！")
                break

    def add_noise(self, x, n_points=1):
        """在 x 中随机翻转 n_points 个像素 (-1 ↔ 1)"""
        x_noisy = x.copy()
        indices = random.sample(range(len(x)), n_points)
        for idx in indices:
            x_noisy[idx] = -x_noisy[idx]
        return x_noisy

    def test_with_noise(self, X, y, n_noise=1, n_trials=10):
        correct = 0
        total = len(X) * n_trials
        for x, label in zip(X, y):
            true_class = np.argmax(label)
            for _ in range(n_trials):
                x_noisy = self.add_noise(x, n_noise)
                pred = self.predict(x_noisy)
                if pred == true_class:
                    correct += 1
        return correct / total
    
samples = [
([ 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 0]),

([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0]
),

([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
[0, 0, 1, 0, 0, 0, 0]),

([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
[0, 0, 0, 0, 0, 0, 1]),

([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 0]),

([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 1, 0, 0]),

([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 0, 0, 0]),

([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
[1, 0, 0, 0, 0, 0, 0]),

([0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0]),

([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 1, 0, 0, 0, 0]),

([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 1]),

([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 0]),

([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 1, 0, 0]),

([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 0, 0, 0]),

([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
[1, 0, 0, 0, 0, 0, 0]),

([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0]),


([0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 1, 0, 0, 0, 0]),

([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
[0, 0, 0, 0, 0, 0, 1]),

([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 0]),

([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 1, 0, 0]),

([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 1, 0, 0, 0]),
]


# 准备数据
X = np.array([s[0] for s in samples])  # (21, 63)
y = np.array([s[1] for s in samples])  # (21, 7)

# ========== 实验一：对比两种训练方式 ==========

print("="*60)
print("实验一：对比仅用原始数据 vs 原始+噪声数据训练")

# 方法一：仅原始数据
model_clean = MultiClassPerceptron(eta=0.5)
print("\n--- 方法一：仅使用原始数据训练 ---")
epochs_clean = model_clean.train(X, y, epochs=20)
acc1_clean = model_clean.test_with_noise(X, y, n_noise=1, n_trials=10)
acc2_clean = model_clean.test_with_noise(X, y, n_noise=2, n_trials=10)
print(f"训练轮数: {epochs_clean}")
print(f"1 个噪声点准确率: {acc1_clean:.2%}")
print(f"2 个噪声点准确率: {acc2_clean:.2%}")

# 方法二：原始数据 + 每个样本加一个噪声点
X_augmented = []
y_augmented = []

for x, label in zip(X, y):
    X_augmented.append(x)
    y_augmented.append(label)
    # 添加一个噪声版本
    x_noisy = model_clean.add_noise(x, n_points=1)
    X_augmented.append(x_noisy)
    y_augmented.append(label)  # 标签不变

X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

model_aug = MultiClassPerceptron(eta=0.5)
print("\n--- 方法二：使用原始数据 + 1个噪声点增强数据训练 ---")
epochs_aug = model_aug.train(X_augmented, y_augmented, epochs=20)
acc1_aug = model_aug.test_with_noise(X, y, n_noise=1, n_trials=10)
acc2_aug = model_aug.test_with_noise(X, y, n_noise=2, n_trials=10)
print(f"训练轮数: {epochs_aug}")
print(f"1 个噪声点准确率: {acc1_aug:.2%}")
print(f"2 个噪声点准确率: {acc2_aug:.2%}")

# 对比结果
print("\n📊 性能对比总结:")
print(f"仅原始数据 -> 1噪准确率: {acc1_clean:.2%}, 2噪准确率: {acc2_clean:.2%}")
print(f"增强数据 -> 1噪准确率: {acc1_aug:.2%}, 2噪准确率: {acc2_aug:.2%}")


# ========== 实验二：对比不同学习率 ==========

print("="*60)
print("实验二：对比不同学习率 (η=0.1, 0.5, 1.0)")

etas = [0.1, 0.5, 1.0]
results = []

for eta in etas:
    print(f"\n--- 学习率 η={eta} ---")
    model_eta = MultiClassPerceptron(eta=eta)
    epochs = model_eta.train(X, y, epochs=20)
    acc1 = model_eta.test_with_noise(X, y, n_noise=1, n_trials=10)
    acc2 = model_eta.test_with_noise(X, y, n_noise=2, n_trials=10)
    results.append({
        'eta': eta,
        'epochs': epochs,
        'acc1': acc1,
        'acc2': acc2
    })
    print(f"训练轮数: {epochs}")
    print(f"1 个噪声点准确率: {acc1:.2%}")
    print(f"2 个噪声点准确率: {acc2:.2%}")

# 绘图：学习率 vs 准确率 & 收敛轮数
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Learning Rate (η)')
ax1.set_ylabel('Accuracy (%)', color='blue')
ax1.plot(etas, [r['acc1']*100 for r in results], 'o-', color='blue', label='1 Noise Acc')
ax1.plot(etas, [r['acc2']*100 for r in results], 's-', color='green', label='2 Noise Acc')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Epochs to Converge', color='red')
ax2.plot(etas, [r['epochs'] for r in results], '^-', color='red', label='Convergence Epochs')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()
plt.title('Impact of Learning Rate on Performance and Convergence')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))
plt.grid(True)
plt.show()