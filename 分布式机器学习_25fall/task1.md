### 实验一：梯度下降单机优化
#### 在 MNIST 数据集上完成图像分类任务

#### 参考 SGD 的实现示例，实现 SGDM、ADAM 两种基于梯度的优化方法，写出两个优化器类
关键代码：
```py
class SGDMOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001, beta1=0.9):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.state = dict()
        for p in self.params:
            self.state[p] = dict()
            self.state[p]['t'] = 0
            self.state[p]['m'] = 0  # 一阶动量
            
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            self.state[p]['t'] += 1
            
            # SGDM公式: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            self.state[p]['m'] = self.beta1 * self.state[p]['m'] + (1 - self.beta1) * p.grad
            
            # 参数更新: w = w - lr * m_t
            p.data.add_(self.state[p]['m'], alpha=-self.lr)
            
class AdamOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = dict()
        for p in self.params:
            self.state[p] = dict()
            self.state[p]['t'] = 0
            self.state[p]['m'] = 0  # 一阶动量
            self.state[p]['V'] = 0  # 二阶动量
            
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            self.state[p]['t'] += 1
            t = self.state[p]['t']
            
            # Adam公式: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            self.state[p]['m'] = self.beta1 * self.state[p]['m'] + (1 - self.beta1) * p.grad
            # Adam公式: V_t = β2 * V_{t-1} + (1 - β2) * (g_t)^2
            self.state[p]['V'] = self.beta2 * self.state[p]['V'] + (1 - self.beta2) * (p.grad ** 2)
            
            # 偏差修正: m_hat_t = m_t / (1 - β1^t)
            m_hat = self.state[p]['m'] / (1 - self.beta1 ** t)
            # 偏差修正: V_hat_t = V_t / (1 - β2^t)
            V_hat = self.state[p]['V'] / (1 - self.beta2 ** t)
            
            # 参数更新: w = w - lr * m_hat_t / (sqrt(V_hat_t) + eps)
            p.data.add_(-self.lr * m_hat / (torch.sqrt(V_hat) + self.eps))

```

#### 绘制两种优化方法下的 loss 函数变化图像；

<img src="../figures_zm/adam_cnn_loss.png" alt="adam_loss" width="600" height="400">
