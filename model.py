#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2023/11/24 18:26
"""
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

dataset = KarateClub()

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)

        self.classifier = nn.Linear(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # (num_nodes, num_features=34) ——> (num_nodes, num_features=64)
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # (num_nodes, num_features=64) ——> (num_nodes, num_features=32)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # (num_nodes, num_features=32) ——> (num_nodes, num_features=16)
        x = F.relu(x)

        x = self.classifier(x)  # (num_nodes, num_features=16) ——> (num_nodes, dataset.num_classes=4)

        return x
