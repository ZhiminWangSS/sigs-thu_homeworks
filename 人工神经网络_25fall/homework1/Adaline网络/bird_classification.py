import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def species_generator(mu1, sigma1, mu2, sigma2, n_samples, target, seed=42):
    """生成两类鸟类数据"""
    np.random.seed(seed)
    f1 = np.random.normal(mu1, sigma1, n_samples)  # 体重
    f2 = np.random.normal(mu2, sigma2, n_samples)  # 翼展
    X = np.array([f1, f2]).T
    y = np.full(n_samples, target)
    return X, y


class ADALINE:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.weights = None

    def train(self, X, y, max_epochs=1000):
        """X: (n, 2), y: (n,)"""
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # (n, 3)
        self.weights = np.zeros(X_bias.shape[1])  # [b, w1, w2]
        mse_history = []

        for epoch in range(max_epochs):
            errors = []
            for i in range(len(X_bias)):
                net = np.dot(self.weights, X_bias[i])
                error = y[i] - net
                self.weights += self.eta * error * X_bias[i]
                errors.append(error**2)

            mse = np.mean(errors)
            mse_history.append(mse)

            if mse < 1e-6:
                print(f"✅ 第 {epoch+1} 轮收敛！MSE = {mse:.6f}")
                break

        return mse_history

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        net = np.dot(X_bias, self.weights)
        return np.sign(net)

    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)


# ========== 数据生成 ==========
X_albatross, y_albatross = species_generator(9000, 800, 300, 20, 100, 1, seed=1)
X_owl, y_owl = species_generator(1000, 200, 100, 15, 100, -1, seed=2)
X = np.vstack([X_albatross, X_owl])
y = np.hstack([y_albatross, y_owl])

print(f"信天翁样本数: {len(X_albatross)}")
print(f"猫头鹰样本数: {len(X_owl)}")

# ========== 特征标准化 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 不同学习率实验 ==========
etas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1, 6e-1, 7e-1, 9e-1, 1.0, 5.0, 10.0]
results = {}

for eta in etas:
    print(f"\n--- 学习率 η={eta} ---")
    model = ADALINE(eta=eta)
    try:
        mse_hist = model.train(X_scaled, y, max_epochs=1000)
        acc = model.accuracy(X_scaled, y)
        epochs = len(mse_hist)
        final_mse = mse_hist[-1]
        results[eta] = {
            "converged": True,
            "epochs": epochs,
            "final_mse": final_mse,
            "accuracy": acc,
            "mse_hist": mse_hist,
        }
        print(f"收敛轮数: {epochs}, 最终MSE: {final_mse:.6f}, 准确率: {acc:.2%}")
    except Exception as e:
        print(f"训练失败: {e}")
        results[eta] = {"converged": False}

# ========== 找出临界学习率 ==========
critical_eta = None
for eta in sorted(results.keys()):
    if not results[eta]["converged"]:
        critical_eta = eta
        break

if critical_eta:
    print(f"\n⚠️ 临界学习率：当 η ≥ {critical_eta} 时，ADALINE 不再收敛！")
else:
    print("\n✅ 所有学习率下均收敛。")

# ========== 绘制误差曲线对比图 ==========
plt.figure(figsize=(12, 5))

# 子图1：所有学习率的 MSE 曲线
plt.subplot(1, 2, 1)
for eta in etas:
    if results[eta]["converged"]:
        mse_hist = results[eta]["mse_hist"]
        plt.plot(range(1, len(mse_hist) + 1), mse_hist, label=f"η={eta}", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("ADALINE Training Error Curves (Different Learning Rates)")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.0)  # 限制Y轴范围便于观察

# 子图2：学习率 vs 收敛性/准确率
plt.subplot(1, 2, 2)
etas_converged = [e for e in etas if results[e]["converged"]]
accuracies = [results[e]["accuracy"] for e in etas_converged]
epochs_to_converge = [results[e]["epochs"] for e in etas_converged]

ax1 = plt.gca()
ax1.bar(
    range(len(etas_converged)),
    accuracies,
    tick_label=etas_converged,
    color="green",
    alpha=0.7,
)
ax1.set_xlabel("Learning Rate (η)")
ax1.set_ylabel("Accuracy (%)", color="green")
ax1.tick_params(axis="y", labelcolor="green")

ax2 = ax1.twinx()
ax2.plot(
    range(len(etas_converged)),
    epochs_to_converge,
    "ro-",
    linewidth=2,
    label="Epochs to Converge",
)
ax2.set_ylabel("Epochs to Converge", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0, max(epochs_to_converge) * 1.1)

plt.title("Performance vs Learning Rate")
plt.tight_layout()
plt.show()

# ========== 绘制最终分类边界 ==========
model_final = ADALINE(eta=1e-4)  # 选择一个稳定的学习率
model_final.train(X_scaled, y, max_epochs=1000)
acc_final = model_final.accuracy(X_scaled, y)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_albatross[:, 0],
    X_albatross[:, 1],
    c="blue",
    marker="o",
    s=50,
    label="Albatross (+1)",
)
plt.scatter(X_owl[:, 0], X_owl[:, 1], c="red", marker="+", s=100, label="Owl (-1)")

# 分类边界（需转换回原始尺度）
b, w1, w2 = model_final.weights
mean1, mean2 = scaler.mean_
std1, std2 = scaler.scale_

x1_vals = np.linspace(500, 12000, 100)
x2_vals = -((w1 / std1) * x1_vals + b - (w1 * mean1 / std1) - (w2 * mean2 / std2)) * (
    std2 / w2
)

plt.plot(x1_vals, x2_vals, "g-", linewidth=2, label="Decision Boundary")

plt.xlabel("Weight (g)")
plt.ylabel("Wingspan (cm)")
plt.title(f"ADALINE Classification (η=1e-4, Acc={acc_final:.2%})")
plt.legend()
plt.grid(True)
plt.axis([500, 12000, 50, 350])
plt.show()
