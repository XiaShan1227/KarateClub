#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2023/11/23 19:38
"""
# "空手道俱乐部"数据集,一张图,节点分类任务
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

import networkx as nx
import matplotlib.pyplot as plt

dataset = KarateClub()

# print(dir(dataset))  # 封装类的属性
# python的f-string用法
print(f"Dataset: {dataset}")  # 数据集名称
print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")  # 数据集有4个类别
print(f"Number of features: {dataset.num_features} \n")  # 每个节点特征维度为34

data = dataset[0]  # 获取第一个图对象

print(data)
# print(dir(data))
# print(data.train_mask)
print(f"Number of nodes: {data.num_nodes}")  # 节点的数量
print(f"Number of edges: {data.num_edges}")  # 边的数量
print(f"Average degree of nodes: {data.num_edges / data.num_nodes:.2f}")  # 节点平均度
print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")  # 是否有孤立节点
print(f"Has self-loops: {data.has_self_loops()}")  # 是否有自循环
print(f"Is undirected: {data.is_undirected()} \n")  # 是否无向图

G = to_networkx(data)  # 转换成networkx.Graph
print(G)

nx.draw_networkx(G, pos=nx.spring_layout(G, seed=16), arrows=False, with_labels=True, node_color=data.y, cmap='cool')
plt.title("Graph of KarateClub")  # 设置标题
plt.savefig("KarateClub.png")  # 保存图片
