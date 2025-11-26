# 分布式训练启动指南

本项目提供了两种启动分布式训练的方式：使用bash脚本或Python脚本。

## 前提条件

- 确保已安装Python和PyTorch
- 确保所有必要的依赖项已安装（torch, torchvision等）
- 确保网络配置允许进程间通信

## 方法一：使用Bash脚本

```bash
# 直接运行脚本
./run_distributed_training.sh
```

该脚本将启动3个进程（rank 0, 1, 2），并将日志保存在`logs`目录中。

## 方法二：使用Python脚本

```bash
# 使用默认参数
python run_distributed_training.py

# 或者自定义参数
python run_distributed_training.py --world_size 3 --master_addr localhost --master_port 12355 --log_dir logs
```

### Python脚本参数说明

- `--world_size`: 总进程数（默认：3）
- `--master_addr`: 主节点地址（默认：localhost）
- `--master_port`: 主节点端口（默认：12355）
- `--log_dir`: 日志目录（默认：logs）

## 日志查看

训练完成后，日志将保存在`logs`目录中，每个进程一个日志文件：
- `rank0.log`: 主节点日志
- `rank1.log`: 工作节点1日志
- `rank2.log`: 工作节点2日志

## 注意事项

1. 确保防火墙设置允许指定端口的通信
2. 如果在多台机器上运行，需要将`master_addr`设置为主节点的IP地址
3. 确保所有节点上的代码和环境一致
4. 如果遇到端口占用问题，可以尝试更改`master_port`参数

## 故障排除

1. 如果进程无法启动，检查Python环境和依赖项
2. 如果进程间通信失败，检查网络配置和防火墙设置
3. 查看日志文件以获取详细错误信息

## 手动启动（可选）

如果需要手动启动各个进程，可以使用以下命令：

```bash
# 终端1：启动主节点
python model.py --n_devices 3 --rank 0 --master_addr localhost --master_port 12355

# 终端2：启动工作节点1
python model.py --n_devices 3 --rank 1 --master_addr localhost --master_port 12355

# 终端3：启动工作节点2
python model.py --n_devices 3 --rank 2 --master_addr localhost --master_port 12355
```