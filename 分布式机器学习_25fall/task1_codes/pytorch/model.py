import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from MyOptimizer import SGDOptimizer, SGDMOptimizer ,AdamOptimizer

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)
        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class MLPNet(nn.Module):
    """
    多层感知机网络，保持与Net类相同的层数，但全部使用全连接层
    """
    def __init__(self, input_size=784, num_classes=10):
        super(MLPNet, self).__init__()
        
        # 第一层：输入层到隐藏层1 (784 -> 6*28*28)
        self.fc1 = nn.Linear(input_size, 6*28*28)
        # 第二层：隐藏层1到隐藏层2 (6*28*28 -> 16*5*5)
        self.fc2 = nn.Linear(6*28*28, 16*5*5)
        # 第三层：隐藏层2到隐藏层3 (16*5*5 -> 120)
        self.fc3 = nn.Linear(16*5*5, 120)
        # 第四层：隐藏层3到输出层 (120 -> num_classes)
        self.fc4 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28) 或 (b, 784)
        """
        # 如果输入是图像格式，先展平
        if x.dim() == 4:
            x = x.flatten(1)
        
        # 第一层：输入层到隐藏层1
        out = F.relu(self.fc1(x))
        # 第二层：隐藏层1到隐藏层2
        out = F.relu(self.fc2(out))
        # 第三层：隐藏层2到隐藏层3
        out = F.relu(self.fc3(out))
        # 第四层：隐藏层3到输出层
        out = self.fc4(out)

        return out

def train(model, dataloader, optimizer, loss_fn, num_epochs=1, writer=None, use_profiler=False):
    print("Start training ...")
    loss_total = 0.
    model.train()
    
    # 创建logs目录
    if writer is None:
        os.makedirs("logs", exist_ok=True)
        writer = SummaryWriter("logs")
    
    global_step = 0
    
    # 初始化Profiler
    profiler = None
    if use_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(writer.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for i, batch_data in enumerate(dataloader):
            step_start_time = time.time()
            
            # 记录训练开始前的显存使用情况
            memory_before = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
            max_memory_before = torch.cuda.max_memory_allocated() / 1024**2
            
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            # 前向传播时间统计
            forward_start = time.time()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            forward_time = (time.time() - forward_start) * 1000  # 转换为毫秒

            # 反向传播时间统计
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_time = (time.time() - backward_start) * 1000

            # 参数更新时间统计
            step_start = time.time()
            optimizer.step()
            step_time = (time.time() - step_start) * 1000
            
            # 记录训练后的显存使用情况
            memory_after = torch.cuda.memory_allocated() / 1024**2
            max_memory_after = torch.cuda.max_memory_allocated() / 1024**2
            
            total_step_time = (time.time() - step_start_time) * 1000
            
            loss_total += loss.item()
            
            # Profiler步骤
            if profiler is not None:
                profiler.step()
            
            # 每20步记录一次TensorBoard数据
            if i % 20 == 19:
                avg_loss = loss_total / 20
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, avg_loss))
                
                # 记录训练loss
                writer.add_scalar('Loss/train', avg_loss, global_step)
                
                # 记录显存使用情况
                writer.add_scalar('Memory/allocated', memory_after, global_step)
                writer.add_scalar('Memory/max_allocated', max_memory_after, global_step)
                
                # 记录时间开销
                writer.add_scalar('Time/forward_pass', forward_time, global_step)
                writer.add_scalar('Time/backward_pass', backward_time, global_step)
                writer.add_scalar('Time/optimizer_step', step_time, global_step)
                writer.add_scalar('Time/total_step', total_step_time, global_step)
                
                loss_total = 0.0
            
            global_step += 1
        
        # 记录每个epoch的总时间
        epoch_time = (time.time() - epoch_start_time) * 1000
        writer.add_scalar('Time/epoch', epoch_time, epoch)
    
    # 停止Profiler
    if profiler is not None:
        profiler.stop()
        print("Profiler数据已保存到TensorBoard")
    
    print("Training Finished!")
    writer.close()

def test(model: nn.Module, test_loader, writer=None, global_step=0):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    test_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            
            # 计算验证集loss
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / size
    
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size, accuracy))
    
    # 记录验证集loss和准确率到TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/validation', test_loss, global_step)
        writer.add_scalar('Accuracy/validation', accuracy, global_step)
    
    return test_loss, accuracy

def main(optimizer_type="SGD", run_name=None, lr=None, beta1=None, beta2=None, use_profiler=False, model_type="CNN"):
    """主函数，支持选择不同的优化器类型和自定义运行名称
    
    Args:
        optimizer_type (str): 优化器类型，支持 "SGD", "SGDM", "Adam"
        run_name (str): TensorBoard运行名称，如果为None则自动生成
        lr (float): 学习率，如果为None则使用默认值
        beta1 (float): 一阶动量衰减率，如果为None则使用默认值
        beta2 (float): 二阶动量衰减率，如果为None则使用默认值
        use_profiler (bool): 是否启用TensorBoard Profiler
    """
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)
    
    # 为每次训练创建唯一的运行名称
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果用户没有提供运行名称，则自动生成
    if run_name is None:
        run_name = f"{optimizer_type}_{model_type}_{timestamp}"
    else:
        run_name = f"{run_name}_{model_type}_{timestamp}"
    
    writer = SummaryWriter(f"logs/{run_name}")
    
    if model_type == "CNN":
        model = Net(in_channels=1, num_classes=10)
    elif model_type == "MLP":
        model = MLPNet(input_size=784, num_classes=10)
    model.cuda()

    DATA_PATH = "./data"

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    
    # 设置默认参数
    if lr is None:
        if optimizer_type == "Adam":
            lr = 0.001
        else:
            lr = 0.01
    
    if beta1 is None:
        beta1 = 0.9
    
    if beta2 is None:
        beta2 = 0.999
    
    # 根据优化器类型选择不同的优化器
    if optimizer_type == "SGD":
        optimizer = SGDOptimizer(model.parameters(), lr=lr)
        print(f"使用SGD优化器，学习率: {lr}")
    elif optimizer_type == "SGDM":
        optimizer = SGDMOptimizer(model.parameters(), lr=lr, beta1=beta1)
        print(f"使用SGDM优化器，学习率: {lr}, 动量系数: {beta1}")
    elif optimizer_type == "Adam":
        optimizer = AdamOptimizer(model.parameters(), lr=lr, beta1=beta1, beta2=beta2)
        print(f"使用Adam优化器，学习率: {lr}, beta1: {beta1}, beta2: {beta2}")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    # 训练模型并传递writer
    train(model, train_loader, optimizer, loss_fn, writer=writer)
    
    # 测试模型并记录验证集指标
    test_loss, accuracy = test(model, test_loader, writer=writer, global_step=0)
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f"TensorBoard日志已保存到 logs/{run_name} 目录")
    print(f"启动TensorBoard服务命令: tensorboard --logdir=logs")

if __name__ == "__main__":  
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    # 示例：可以在这里指定不同的优化器和运行名称
    # main(optimizer_type="SGD", run_name="my_sgd_experiment")
    # main(optimizer_type="Adam", run_name="my_adam_experiment")
    main(optimizer_type="SGD")