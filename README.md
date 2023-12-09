# GTCAlign: Global Topology Consistency-based Graph Alignment
This is a pytorch implementation of GTCAlign, as described in our paper:
GTCAlign: Global Topology Consistency-based Graph Alignment (TKDE 2023)

## Requirements
- pytorch >= 1.10.1+cu113
- networkx == 1.11.0
- pyg == 2.0.4

## Run the demo
```python
python main.py
```

## Dataset

Here we have uploaded two datasets: the Douban and PPI datasets, and more descriptions of the datasets can be found by moving to our graph alignment framework (with subsequent updates to the github links).

## Cite
```python
@ARTICLE {10241993,
author = {C. Wang and P. Jiang and X. Zhang and P. Wang and T. Qin and X. Guan},
journal = {IEEE Transactions on Knowledge &amp; Data Engineering},
title = {GTCAlign: Global Topology Consistency-based Graph Alignment},
year = {5555},
volume = {},
number = {01},
issn = {1558-2191},
pages = {1-16},
abstract = {Graph alignment aims to find correspondent nodes between two graphs. Most existing algorithms assume that correspondent nodes in different graphs have similar local structures. However, this principle may not apply to some real-world application scenarios when two graphs have different densities. Some correspondent node pairs may have very different local structures in these cases. Nevertheless, correspondent nodes are expected to have similar importance, inspiring us to exploit global topology consistency for graph alignment. This paper presents GTCAlign, an unsupervised graph alignment framework based on global topology consistency. An indicating matrix is calculated to show node pairs with consistent global topology based on a comprehensive centrality metric. A graph convolutional network (GCN) encodes local structural and attributive information into low-dimensional node embeddings. Then, node similarities are computed based on the obtained node embeddings under the guidance of the indicating matrix. Moreover, a pair of nodes are more likely to be aligned if most of their neighbors are aligned, motivating us to develop an iterative algorithm to refine the alignment results recursively. We conduct extensive experiments on real-world and synthetic datasets to evaluate the effectiveness of GTCAlign. The experimental results show that GTCAlign outperforms state-of-the-art graph alignment approaches.},
keywords = {topology;network topology;representation learning;iterative methods;social networking (online);training;task analysis},
doi = {10.1109/TKDE.2023.3312358},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {sep}
}
```
