import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import argparse
import time
import torch.distributed as dist 
import torch.multiprocessing as mp
import dist_utils
from sampler import MySampler, RandomSampler, RandomSplitSampler
from torch.utils.tensorboard import SummaryWriter

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


def train(model, dataloader, loss_fn, optimizer, num_epochs=2, writer=None, epoch=0):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    model.train()

    # 初始化模型参数，确保所有进程有相同的初始参数
    dist_utils.init_parameters(model)

    for i, batch_data in enumerate(dataloader):
        batch_start_time = time.time()
        inputs, labels = batch_data
        # 确保数据移动到正确的GPU设备
        device = torch.device(f"cuda:{dist_utils.get_local_rank()}")
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # 记录通信前时间
        comm_start_time = time.time()
        # averge the gradients of model parameters
        dist_utils.average_gradients(model)
        comm_time = time.time() - comm_start_time
        
        optimizer.step()
        batch_time = time.time() - batch_start_time
        
        loss_total += loss.item()
        train_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)
        
        if writer and i % 20 == 19:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Time/batch_time', batch_time, epoch * len(dataloader) + i)
            writer.add_scalar('Time/communication_time', comm_time, epoch * len(dataloader) + i)
        
        if i % 20 == 19:    
            print('Device: %d epoch: %d, iters: %5d, loss: %.3f' % (dist_utils.get_local_rank(), epoch + 1, i + 1, loss_total / 20))
            loss_total = 0.0

    train_loss /= len(dataloader)
    accuracy = 100. * correct / total
    epoch_time = time.time() - start_time
    
    if writer:
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('Time/epoch_time', epoch_time, epoch)
    
    print(f"Training Finished! Average Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    return train_loss, accuracy, epoch_time

def test(model: nn.Module, test_loader, writer=None, epoch=0):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    test_loss = 0
    start_time = time.time()
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 确保数据移动到正确的GPU设备
            device = torch.device(f"cuda:{dist_utils.get_local_rank()}")
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            test_loss += F.cross_entropy(output, labels, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    
    test_loss /= size
    accuracy = 100. * correct / size
    test_time = time.time() - start_time
    
    if writer:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        writer.add_scalar('Time/test_time', test_time, epoch)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Time: {:.2f}s\n'.format(
        test_loss, correct, size, accuracy, test_time))
    return test_loss, accuracy, test_time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--gpu', default="0", type=str, help='GPU ID')
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')
    parser.add_argument('--sampler_type', default='randomsampler', type=str, choices=['randomsampler', 'randomsplitsampler'], help='Sampler type for data loading')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 2)')
    parser.add_argument('--log_dir', type=str, default='./logs', help='directory for tensorboard logs (default: ./logs)')

    args = parser.parse_args()
    return args

def main(args=None):
    if args is None:
        args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DATA_PATH = "./data"
    # initialize process group
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    # construct the model
    model = Net(in_channels=1, num_classes=10)
    device = torch.device(f"cuda:{dist_utils.get_local_rank()}")
    model = model.to(device)

    # construct the dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    # 根据采样器类型选择不同的采样器
    if args.sampler_type == 'mysampler':
        sampler = MySampler(train_set, args.n_devices, args.rank, shuffle=True, seed=args.rank)
    elif args.sampler_type == 'randomsampler':
        sampler = RandomSampler(train_set, shuffle=True, seed=args.rank)
    elif args.sampler_type == 'randomsplitsampler':
        sampler = RandomSplitSampler(train_set, args.n_devices, args.rank, shuffle=True, seed=args.rank)
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler_type}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # construct the loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 创建TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        train_loss, train_acc, train_time = train(model, train_loader, loss_fn, optimizer, args.epochs, writer, epoch)
        test_loss, test_acc, test_time = test(model, test_loader, writer, epoch)
    
    total_time = time.time() - total_start_time
    
    if writer:
        writer.add_scalar('Time/total_training_time', total_time)
        writer.close()
    
    print(f'Total training time: {total_time:.2f}s')

if __name__ == "__main__":
    main()