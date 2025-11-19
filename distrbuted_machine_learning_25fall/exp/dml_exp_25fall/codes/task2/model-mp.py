import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import argparse
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import dist_utils
import matplotlib.pyplot as plt
import time
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


def train(model, dataloader, loss_fn, optimizer, num_epochs=2, comm_method="allreduce", writer=None):
    print("Device {} starts training with {} communication method ...".format(dist_utils.get_local_rank(), comm_method))
    loss_total = 0.
    model.train()

    # sync the paramters: make models in different nodes the same.
    dist_utils.init_parameters(model)

    starttime = time.time()
    comm_time_total = 0.0
    comm_count = 0
    
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            
            # Measure communication time for gradient averaging
            comm_start = time.time()
            if comm_method == "allreduce_avg_sum":
                if epoch < num_epochs/2:
                    dist_utils.allreduce_average_gradients(model)
                else:
                    dist_utils.allreduce_sum_gradients(model)
            elif comm_method == "allreduce_sum_avg":
                if epoch < num_epochs/2:
                    dist_utils.allreduce_sum_gradients(model)
                else:
                    dist_utils.allreduce_average_gradients(model)
            elif comm_method == "allreduce":
               dist_utils.allgather_average_gradients(model) 
            elif comm_method == "allgather":
                dist_utils.allgather_average_gradients(model)
            elif comm_method == "reduce":
                dist_utils.reduce_average_gradients(model)
            comm_end = time.time()
            comm_time_total += (comm_end - comm_start)
            comm_count += 1

            optimizer.step()
            loss_total += loss.item()

            # Log each batch loss to TensorBoard (only rank 0)
            if writer is not None and dist_utils.get_local_rank() == 0:
                writer.add_scalar('Loss/batch_loss', loss.item(), epoch * len(dataloader) + i)

            if i % 20 == 19:    
                avg_loss = loss_total / 20
                print('Device: %d epoch: %d, iters: %5d, loss: %.3f, comm_time: %.6f' % 
                      (dist_utils.get_local_rank(), epoch + 1, i + 1, avg_loss, comm_time_total / comm_count))
                
                # Log to TensorBoard (only rank 0)
                if writer is not None and dist_utils.get_local_rank() == 0:
                    writer.add_scalar(f'Loss/loss', avg_loss, epoch * len(dataloader) + i)
                    writer.add_scalar(f'Comm_time/comm_time', comm_time_total / comm_count, epoch * len(dataloader) + i)
                
                loss_total = 0.0

    print("Training Finished!")
    endtime = time.time()
    train_time = endtime-starttime
    avg_comm_time = comm_time_total / comm_count if comm_count > 0 else 0.0
    print("Training time: {}, Avg communication time: {}".format(train_time, avg_comm_time))
    
    return avg_comm_time


def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--gpu', default="0", type=str, help='GPU ID')
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')
    parser.add_argument('--comm_method', default='allreduce', type=str, choices=['allreduce', 'allgather', 'reduce','allreduce_avg_sum','allreduce_sum_avg'], 
                        help='Communication method for gradient averaging')

    args = parser.parse_args()

    return args


def main(rank, args):
    args.rank=rank

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank) # args.gpu
    DATA_PATH = "./data"
    os.makedirs(DATA_PATH, exist_ok=True)
    # initialize process group
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    
    # Initialize TensorBoard writer (only rank 0)
    writer = None
    if dist_utils.get_local_rank() == 0:
        writer = SummaryWriter(log_dir=f'./runs/{args.comm_method}')
    
    # construct the model
    model = Net(in_channels=1, num_classes=10)
    model.cuda()  


    # construct the dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    sampler = DistributedSampler(dataset=train_set, num_replicas=args.n_devices, rank=args.rank)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # construct the loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train with specified communication method
    avg_comm_time = train(model, train_loader, loss_fn, optimizer, comm_method=args.comm_method, writer=writer)
    
    # Test and record accuracy
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    
    accuracy = 100 * correct / size
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, size, accuracy))
    
    # Log test accuracy to TensorBoard (only rank 0)
    if writer is not None and dist_utils.get_local_rank() == 0:
        writer.add_scalar('Accuracy/test', accuracy, 0)
        writer.add_scalar('Comm_time/avg', avg_comm_time, 0)
        writer.close()
    
    # Clean up distributed process group
    dist_utils.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(main, (args,), nprocs=args.n_devices)