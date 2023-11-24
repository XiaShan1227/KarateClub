#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2023/11/24 18:29
"""
import torch, random, argparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import KarateClub

from model import GCN

def parse_args():
    parser = argparse.ArgumentParser(description='KarateClub Classification(Node Classification)') # 创建ArgumentParser对象
    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment') # 实验随机种子
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--init_lr', type=float, default=0.1, help='Learning rate initialization')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs of training')

    args = parser.parse_args() # 解析命令行参数
    return args


def train(args, data):
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index)) # 使用GPU or CPU
    model = GCN().to(device) # 加载模型
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-4) # 优化器
    criterion = nn.CrossEntropyLoss(reduction='mean')  # 损失函数

    for epoch in range(args.epochs):
        data = data.to(device)

        optimizer.zero_grad()  # 梯度清0
        outputs = model(data.x, data.edge_index)  # 前向传播
        loss = criterion(outputs, data.y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        preds = outputs.max(dim=1)[1]  # 预测标签(outputs.max返回的是两个张量:数值和标签)

        accuracy = (data.y.detach().cpu()==preds.detach().cpu()).sum().item() / len(data.y)

        print(f"Train_epoch: {epoch} ————> Loss: {loss.item():.6f}; Accuracy: {accuracy * 100:.3f}%")


if __name__ == "__main__":
    dataset = KarateClub()
    data = dataset[0]

    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子

    train(args, data)
