import numpy as np
import matplotlib.pyplot as plt

# 数据
xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
ddim = [-1, 1, -1, -1, 1, 1]

def perceptron_train(x_data, d_data, eta=0.5, max_epochs=100):
    w = np.array([0.0, 0.0, 0.0])
    for epoch in range(max_epochs):
        errors = 0
        for x, d in zip(x_data, d_data):
            x_vec = np.array([1, x[0], x[1]])
            net = np.dot(w, x_vec)
            o = 1 if net >= 0 else -1
            if o != d:
                w += eta * (d - o) * x_vec
                errors += 1
        if errors == 0:
            print(f"η={eta}: 第 {epoch + 1} 轮收敛")
            break
    return w

# 训练
w_05 = perceptron_train(xdim, ddim, eta=0.5)
w_10 = perceptron_train(xdim, ddim, eta=1.0)

# 绘图
plt.figure(figsize=(8, 6))
for x, d in zip(xdim, ddim):
    plt.scatter(x[0], x[1], c='blue' if d == 1 else 'red',
                marker='o' if d == 1 else '+', s=100)

x1_vals = np.linspace(-0.8, 0.8, 100)
for w, style, label in [(w_05, 'g-', 'η=0.5'), (w_10, 'k--', 'η=1.0')]:
    b, w1, w2 = w
    x2_vals = -(w1 * x1_vals + b) / w2
    plt.plot(x1_vals, x2_vals, style, linewidth=2, label=label)

plt.xlabel('X1'); plt.ylabel('X2')
plt.title('Perceptron Decision Boundaries with Different Learning Rates')
plt.legend(); plt.grid(True); plt.axis([-0.8, 0.8, -0.25, 1])
plt.savefig('perceptron_boundaries.png')
plt.show()
