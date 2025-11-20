import math
import torch
from torch.utils.data import Dataset, Sampler

class MySampler(Sampler):
    def __init__(self, dataset:Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas    # number of clients (processes)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed                    # set seed to be the rank of the client, to avoid generating the same indice lists.
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas) 

    def __iter__(self):
        """
            example:
                indices=list(range(len(self.dataset)))
                return iter(indices)
        """
        # write your code here
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # 使用固定种子确保不同进程得到不同的随机序列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        
        # 为当前进程分配数据索引
        start_idx = self.rank * self.num_samples
        end_idx = min(start_idx + self.num_samples, len(self.dataset))
        process_indices = indices[start_idx:end_idx]
        
        return iter(process_indices)

    def __len__(self):
        return self.num_samples


class RandomSampler(Sampler):
    """
    每个 epoch 对全集数据随机打乱，然后全部返回。
    不做分布式划分（每个 rank 拿相同的随机序列）。
    """
    def __init__(self, dataset: Dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # 每个 epoch 生成不同随机序列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class RandomSplitSampler(Sampler):
    """
    随机划分采样：
    - 先对全集随机打乱
    - 再按 num_replicas 划分为 num_replicas 份
    - 每个 rank 拿自己的那一份
    """
    def __init__(self, dataset: Dataset, num_replicas, rank, shuffle=True, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # --- 按 rank 划分 ---
        start = self.rank * self.num_samples
        end = min(start + self.num_samples, len(self.dataset))
        split_indices = indices[start:end]

        return iter(split_indices)

    def __len__(self):
        return self.num_samples
