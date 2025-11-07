import numpy as np
import matplotlib.pyplot as plt

# 样本数据（严格按照表1-2）
xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
ddim = [-1, 1, -1, -1, 1, 1]
eta = 0.5  # 学习率


def adaline_train(x_data, d_data, eta=0.5, max_epochs=100):
    """
    ADALINE (LMS) 训练函数
    返回：最终权重 [b, w1, w2]，以及每轮的均方误差（MSE）
    """
    w = np.array([0.0, 0.0, 0.0])  # [b, w1, w2]
    mse_history = []

    for epoch in range(max_epochs):
        errors = []
        for x, d in zip(x_data, d_data):
            x_vec = np.array([1, x[0], x[1]])  # [bias, x1, x2]
            y_hat = np.dot(w, x_vec)  # 线性输出
            error = d - y_hat
            w = w + eta * error * x_vec
            errors.append(error**2)

        mse = np.mean(errors)
        mse_history.append(mse)

        if mse < 1e-5:  # 收敛阈值
            print(f"✅ 在第 {epoch + 1} 轮后收敛！MSE = {mse:.6f}")
            break

    return w, mse_history


w_adaline, mse_hist = adaline_train(xdim, ddim, eta=0.5)

print("ADALINE 最终权重 (b, w1, w2):", w_adaline)

# 绘图
plt.figure(figsize=(8, 6))

# 绘制样本点
for x, d in zip(xdim, ddim):
    if d == 1:
        plt.scatter(
            x[0],
            x[1],
            c="blue",
            marker="o",
            s=100,
            label="Class +1" if x == xdim[0] else "",
        )
    else:
        plt.scatter(
            x[0],
            x[1],
            c="red",
            marker="+",
            s=100,
            label="Class -1" if x == xdim[0] else "",
        )

# 已有 w_adaline = [b, w1, w2]
b, w1, w2 = w_adaline

# 生成网格点
x1_grid = np.linspace(-0.8, 0.8, 100)
x2_grid = np.linspace(-0.25, 1.0, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# 计算函数值 f(x1, x2) = w1*x1 + w2*x2 + b
F = w1 * X1 + w2 * X2 + b

# 绘制等高线图（特别关注 f=0 的那条线）
plt.figure(figsize=(10, 8))
contour = plt.contour(
    X1, X2, F, levels=[0], colors="green", linewidths=3, label="f(x1,x2)=0"
)
plt.clabel(contour, inline=True, fontsize=12)

# 绘制样本点
for x, d in zip(xdim, ddim):
    if d == 1:
        plt.scatter(
            x[0],
            x[1],
            c="blue",
            marker="o",
            s=100,
            label="Class +1" if x == xdim[0] else "",
        )
    else:
        plt.scatter(
            x[0],
            x[1],
            c="red",
            marker="+",
            s=100,
            label="Class -1" if x == xdim[0] else "",
        )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title(
    "ADALINE Linear Function: f(x1, x2) = w1·x1 + w2·x2 + b\nClassification Boundary: f(x1,x2)=0"
)
plt.legend()
plt.grid(True)
plt.axis([-0.8, 0.8, -0.25, 1])
plt.gca().set_aspect("equal", adjustable="box")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# 生成网格
x1_grid = np.linspace(-0.8, 0.8, 50)
x2_grid = np.linspace(-0.25, 1.0, 50)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# 计算 f(x1, x2)
F = w1 * X1 + w2 * X2 + b

# 绘制曲面
surf = ax.plot_surface(X1, X2, F, cmap="viridis", alpha=0.8, edgecolor="none")

# 标注颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="f(x1, x2)")

# 绘制样本点（投影到曲面上）
for x, d in zip(xdim, ddim):
    y_val = w1 * x[0] + w2 * x[1] + b
    color = "blue" if d == 1 else "red"
    ax.scatter(x[0], x[1], y_val, c=color, marker="o" if d == 1 else "+", s=100)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("f(x1, x2)")
ax.set_title("3D View of ADALINE Linear Function:\nf(x1, x2) = w1·x1 + w2·x2 + b")
ax.view_init(elev=20, azim=45)  # 调整视角
plt.show()
