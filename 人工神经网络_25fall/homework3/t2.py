import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 定义三个字母 C, H, L 的 5×5 点阵图（使用 0/1）
# ------------------------------------------------------------

C_data = np.array([
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,0],
    [1,0,0,0,1],
    [0,1,1,1,0]
]).flatten()

H_data = np.array([
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,0,0,1]
]).flatten()

L_data = np.array([
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,1,1,1,1]
]).flatten()

x_data = np.array([C_data, H_data, L_data]).astype(np.float32)

# ------------------------------------------------------------
# 生成噪声样本（Hamming 距离 = 1）
# ------------------------------------------------------------
def make_noise_sample(vec):
    noisy = vec.copy()
    idx = random.randint(0, 24)
    noisy[idx] = 1 - noisy[idx]
    return noisy

# 每个字母 5 个噪声样本（训练用）
noise_samples = []
for v in x_data:
    for _ in range(5):
        noise_samples.append(make_noise_sample(v))
noise_samples = np.array(noise_samples)

# 训练集：正确样本 + 15 个噪声样本
train_set = np.vstack([x_data, noise_samples])


# ------------------------------------------------------------
# 胜者为王 WTA
# ------------------------------------------------------------
def WTA(x, w):
    d = np.sum((w - x)**2, axis=1)
    return np.argmin(d)

# ------------------------------------------------------------
# 绘制权向量（5x5）
# ------------------------------------------------------------
def plot_weights(w, title):
    plt.figure(figsize=(10, 3))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(w[i].reshape(5,5), cmap='gray_r')
        plt.title(f"Neuron {i}")
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig('./f2.png')


# ------------------------------------------------------------
# 初始化权重（0~1 随机）
# ------------------------------------------------------------

W = np.random.rand(3, 25)
plot_weights(W, "Initial Weights")

# ------------------------------------------------------------
# 训练参数
# ------------------------------------------------------------

epochs = 100
eta0 = 0.5    # 初始学习率

print("Start training...\n")

# ------------------------------------------------------------
# 训练迭代（真正含循环）
# ------------------------------------------------------------
for ep in range(epochs):
    # 学习率线性下降
    eta = eta0 * (1 - ep / epochs)

    # 每 epoch 需要重新随机打乱训练集
    np.random.shuffle(train_set)

    # 对每个样本进行 WTA 更新
    for x in train_set:
        win = WTA(x, W)
        W[win] += eta * (x - W[win])

    if (ep+1) % 10 == 0:
        print(f"Epoch {ep+1}/{epochs}, eta={eta:.4f}")

print("\nTraining finished.")
plot_weights(W, f"Final Weights after {epochs} epochs")

# ------------------------------------------------------------
# 输出训练后的三个神经元内星向量结果
# ------------------------------------------------------------
print("\n=== Final Weight Vectors (内星向量) ===")
for i in range(3):
    print(f"Neuron {i} ({letters[i]}):")
    print(W[i].reshape(5,5))
    print()


# ------------------------------------------------------------
# 识别剩余的 20 个噪声样本
# ------------------------------------------------------------

def make_all_noise(vec):
    """生成全部 25 个 Hamming 距离=1 的样本"""
    res = []
    for i in range(25):
        tmp = vec.copy()
        tmp[i] = 1 - tmp[i]
        res.append(tmp)
    return np.array(res)

print("\n=== Recognition Test on Remaining Noise Samples ===")

letters = ["C", "H", "L"]

for li, vec in enumerate(x_data):
    all_noise = make_all_noise(vec)
    test_noise = all_noise[5:]  # 去掉前 5 个训练用的噪声

    correct = 0
    for s in test_noise:
        win = WTA(s, W)
        if win == li:
            correct += 1

    print(f"{letters[li]}: accuracy = {correct}/20 = {correct/20:.2f}")
