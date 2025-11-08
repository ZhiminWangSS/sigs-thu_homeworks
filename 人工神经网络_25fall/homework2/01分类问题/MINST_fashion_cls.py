import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import struct

# ===========================
# 1. 自定义MNIST数据加载函数
# ===========================
def load_mnist_images(filename):
    """加载MNIST图像数据"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        images = images.astype(np.float32) / 255.0  # 归一化到 [0,1]
        return images

def load_mnist_labels(filename):
    """加载MNIST标签数据"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# 加载训练和测试数据
train_images = load_mnist_images('../datasets/MNIST/train-images-idx3-ubyte')
train_labels = load_mnist_labels('../datasets/MNIST/train-labels-idx1-ubyte')
test_images = load_mnist_images('../datasets/MNIST/t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('../datasets/MNIST/t10k-labels-idx1-ubyte')

# 转换为PyTorch张量并创建数据集
train_images_tensor = torch.from_numpy(train_images).unsqueeze(1)  # 添加通道维度
train_labels_tensor = torch.from_numpy(train_labels).long()
test_images_tensor = torch.from_numpy(test_images).unsqueeze(1)
test_labels_tensor = torch.from_numpy(test_labels).long()

# 数据预处理
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1,1]
])

# 应用变换
train_images_tensor = transform(train_images_tensor)
test_images_tensor = transform(test_images_tensor)

# 创建数据集
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ===========================
# 2. 定义BP网络结构
# ===========================
class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)   # 输入层->隐层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)    # 隐层->输出层
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # CrossEntropyLoss包含Softmax

# ===========================
# 3. 初始化网络与优化器
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BPNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===========================
# 4. 网络训练
# ===========================
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ===========================
# 5. 测试与准确率评估
# ===========================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
