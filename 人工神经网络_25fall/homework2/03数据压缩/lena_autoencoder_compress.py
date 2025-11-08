import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 设置中文字体和负号显示
import matplotlib
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 加载 Lena 图像
# ------------------------------------------------------------
def load_lena(filename):
    img = Image.open(filename).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    return img

# ------------------------------------------------------------
# 定义 AutoEncoder
# ------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ------------------------------------------------------------
# 训练函数
# ------------------------------------------------------------
def train_autoencoder(model, X, num_epochs=500, lr=0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X = X.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()
    return model, criterion(model(X), X).item()

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    file_path = "lena.png"
    if not os.path.exists(file_path):
        print("❌ lena.png not found, please place it in the current directory.")
        return

    # 加载图像
    lena_img = load_lena(file_path)
    img_height, img_width = lena_img.shape
    print(f"Original image shape: {lena_img.shape}")

    # 将每行作为一个样本
    X = torch.tensor(lena_img, dtype=torch.float32)  # shape: (512, 512)

    # 隐层节点数列表
    hidden_nodes_list = [16, 32, 64, 128, 256]
    mse_list = []
    reconstructions = {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)

    for hn in hidden_nodes_list:
        print(f"🔹 Training AutoEncoder with hidden nodes = {hn}")
        model = AutoEncoder(input_dim=img_width, hidden_dim=hn)
        model.to(device)
        model, mse = train_autoencoder(model, X, num_epochs=500, lr=0.01)
        mse_list.append(mse)
        
        with torch.no_grad():
            reconstructions[hn] = model(X).cpu().numpy()  # X 已经在 GPU
        print(f"Hidden nodes {hn}: Reconstruction MSE = {mse:.6f}")

    # ------------------------------------------------------------
    # 绘制原图与重建图
    # ------------------------------------------------------------
    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(hidden_nodes_list)+1, 1)
    plt.imshow(lena_img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    for i, hn in enumerate(hidden_nodes_list):
        plt.subplot(1, len(hidden_nodes_list)+1, i+2)
        plt.imshow(reconstructions[hn], cmap='gray')
        plt.title(f"Hidden={hn}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("lena_reconstruction_pytorch.png")
    plt.close()

    # ------------------------------------------------------------
    # 绘制 MSE vs 隐层节点数
    # ------------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(hidden_nodes_list, mse_list, 'o-', linewidth=2)
    plt.xlabel("Hidden Nodes")
    plt.ylabel("Reconstruction MSE")
    plt.title("Hidden Nodes vs MSE")
    plt.grid(True)
    plt.savefig("lena_mse_vs_hidden.png")
    plt.close()

    print("\n✅ Results saved:")
    print("   - lena_reconstruction_pytorch.png")
    print("   - lena_mse_vs_hidden.png")

# ------------------------------------------------------------
# 程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
