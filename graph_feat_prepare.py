import os
import sys
import numpy as np
import pandas as pd
import config
import dgl
import torch
from dgl.data import DGLDataset
import copy


# 移动平均平滑函数
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


# 节点特征向量读取函数
def read_ca_data(filepath):
    # 判断数据文件是否存在并且读入文件
    if not os.path.exists(filepath):
        print('The data does not exist!')
        sys.exit(1)
    csv_dataframe = pd.read_csv(filepath)
    csv_np = csv_dataframe.values.astype(np.float32)
    csv_np = csv_np[50:, :]  # 删除前50行不稳定的数据
    csv_attr = csv_dataframe.columns.tolist()

    # 数据预处理（归一化、平滑）
    for col_idx in range(csv_np.shape[1]):
        csv_np[:, col_idx] = moving_average(csv_np[:, col_idx], config.filter_size)
        min_vals = np.amin(csv_np[:, col_idx], axis=0)
        max_vals = np.amax(csv_np[:, col_idx], axis=0)
        csv_np[:, col_idx] = ((csv_np[:, col_idx] - min_vals) / (max_vals - min_vals))

    # 完成数据准备
    print('Reading cadata is well done!')
    return csv_np, csv_attr


# 节点连接关系读取函数
def read_adj_mat(filepath):
    # 判断数据文件是否存在并且读入文件
    if not os.path.exists(filepath):
        print('The data does not exist!')
        sys.exit(1)
    csv_dataframe = pd.read_csv(filepath)
    csv_np = csv_dataframe.values[:, 1:]
    csv_attr = csv_dataframe.columns.tolist()
    csv_attr = csv_attr[1:]

    # 完成数据准备
    print('Reading adjmat is well done!')
    return csv_np, csv_attr


# 根据钙活性数据筛选节点
def select_adj_mat(raw_adj_mat, raw_adj_mat_attr, ca_attr):
    # 判断ca_attr是否隶属于raw_mat_attr
    # 注意：列表元素是有顺序的，集合元素是没有顺序的，此处可以判断隶属关系，但不能依据此顺序选择连接关系矩阵
    if not set(ca_attr).issubset(set(raw_adj_mat_attr)):
        print('Please check the neuron name!')
        print(set(ca_attr) - set(raw_adj_mat_attr))
        sys.exit(1)

    attr_idx_list = []
    for idx in range(len(ca_attr)):
        attr_idx = raw_adj_mat_attr.index(ca_attr[idx])
        attr_idx_list.append(attr_idx)

    selected_adj_mat = raw_adj_mat[attr_idx_list, :]
    selected_adj_mat = selected_adj_mat[:, attr_idx_list]

    # 暂时不考虑自环连接
    node_num = len(ca_attr)
    selected_adj_mat = selected_adj_mat * (np.ones((node_num, node_num)) - np.eye(node_num))

    # 完成节点筛选
    print('Selecting node is well done!')
    print(ca_attr)
    print(selected_adj_mat)
    return selected_adj_mat


# 创建图并且为节点添加特征向量
def gen_graph_feat(adjmat, feature):
    # 检查节点数量与特征向量是否匹配
    rows, cols = adjmat.shape
    seq_len, num_node = feature.shape
    if (rows != cols) or (rows != num_node):
        print('The data does not match!')
        sys.exit(1)

    # 创建图
    src_list, dst_list = [], []
    for row_idx in range(rows):
        for col_idx in range(cols):
            if adjmat[row_idx, col_idx] == 1:
                src_list.append(row_idx)
                dst_list.append(col_idx)
    src_th = torch.tensor(src_list)
    dst_th = torch.tensor(dst_list)
    g = dgl.graph((src_th, dst_th), num_nodes=num_node)

    # 添加特征向量
    g.ndata['feat'] = torch.from_numpy(feature.T)
    print('Creating graph is well done!')
    return g


class NeuCalDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="neuron_calcium")
        self.raw_adj_mat, self.raw_adj_mat_attr = read_adj_mat(config.adjmat_filepath)
        self.ca_data, self.ca_data_attr = read_ca_data(config.cadata_filepath)
        self.selected_adj_mat = select_adj_mat(self.raw_adj_mat, self.raw_adj_mat_attr, self.ca_data_attr)
        self.graph = gen_graph_feat(self.selected_adj_mat, self.ca_data)

    def process(self):
        pass

    def __getitem__(self, item):
        return self.graph

    def __len__(self):
        return 1


# 以边为中心划分数据集（针对transductive link prediction场景）
def split_train_val_test_with_edge(g, adj):
    # 把实际存在的边划分为训练集、测试集、验证集
    pos_u, pos_v = g.edges()
    pos_eids = np.arange(g.num_edges())
    pos_eids = np.random.permutation(pos_eids)
    val_size = int(len(pos_eids) * config.val_ratio)
    tes_size = int(len(pos_eids) * config.tes_ratio)
    tra_size = len(pos_eids) - val_size - tes_size
    val_pos_u, val_pos_v = pos_u[pos_eids[:val_size]], pos_v[pos_eids[:val_size]]
    tes_pos_u, tes_pos_v = pos_u[pos_eids[val_size+tra_size:]], pos_v[pos_eids[val_size+tra_size:]]
    tra_pos_u, tra_pos_v = pos_u[pos_eids[val_size:val_size+tra_size]], pos_v[pos_eids[val_size:val_size+tra_size]]

    # 把实际不存在的边划分为训练集、测试集、验证集
    adj_neg = 1 - adj - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    tes_neg_u, tes_neg_v = neg_u[neg_eids[val_size+tra_size:]], neg_v[neg_eids[val_size+tra_size:]]
    tra_neg_u, tra_neg_v = neg_u[neg_eids[val_size:val_size+tra_size]], neg_v[neg_eids[val_size:val_size+tra_size]]

    # 创建graph
    tra_g = dgl.remove_edges(g, np.concatenate((pos_eids[:val_size], pos_eids[val_size+tra_size:]), axis=0))
    tra_pos_g = dgl.graph((tra_pos_u, tra_pos_v), num_nodes=g.num_nodes())
    tra_neg_g = dgl.graph((tra_neg_u, tra_neg_v), num_nodes=g.num_nodes())
    tes_pos_g = dgl.graph((tes_pos_u, tes_pos_v), num_nodes=g.num_nodes())
    tes_neg_g = dgl.graph((tes_neg_u, tes_neg_v), num_nodes=g.num_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.num_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.num_nodes())

    print('Splitting graph is well done!')
    return tra_g, tra_pos_g, tra_neg_g, val_pos_g, val_neg_g, tes_pos_g, tes_neg_g


def split_edges_with_five_folds(g, adj, num_folds=10):

    # 统计邻接矩阵中存在的和不存在的边
    edges_1 = np.column_stack(np.where(adj == 1))
    edges_0 = np.column_stack(np.where(adj == 0))

    # 均匀分组
    num_edges_1 = edges_1.shape[0]  # 存在的边必须远少于不存在的边
    np.random.shuffle(edges_1)
    np.random.shuffle(edges_0)
    edges_1_groups = np.array_split(edges_1, num_folds)
    edges_0_groups = np.array_split(edges_0[:num_edges_1], num_folds)

    # 构造子集
    g_without_edges = dgl.remove_edges(g, np.arange(g.num_edges()))
    g_tra_list = [copy.deepcopy(g_without_edges) for _ in range(num_folds)]
    g_tra_pos_list = [copy.deepcopy(g_without_edges) for _ in range(num_folds)]
    g_tra_neg_list = [copy.deepcopy(g_without_edges) for _ in range(num_folds)]
    g_val_pos_list = [copy.deepcopy(g_without_edges) for _ in range(num_folds)]
    g_val_neg_list = [copy.deepcopy(g_without_edges) for _ in range(num_folds)]
    for i in range(num_folds):
        edges_1_groups_tmp = [edges_1_groups[j] for j in range(num_folds) if j != i]
        edges_1_groups_tmp_merged = np.concatenate(edges_1_groups_tmp)
        g_tra_list[i].add_edges(edges_1_groups_tmp_merged[:, 0], edges_1_groups_tmp_merged[:, 1])  # 训练图
        g_tra_pos_list[i].add_edges(edges_1_groups_tmp_merged[:, 0], edges_1_groups_tmp_merged[:, 1])  # 训练图pos

        edges_0_groups_tmp = [edges_0_groups[j] for j in range(num_folds) if j != i]
        edges_0_groups_tmp_merged = np.concatenate(edges_0_groups_tmp)
        g_tra_neg_list[i].add_edges(edges_0_groups_tmp_merged[:, 0], edges_0_groups_tmp_merged[:, 1])  # 训练图neg
    
        g_val_pos_list[i].add_edges(edges_1_groups[i][:, 0], edges_1_groups[i][:, 1])  # 测试图pos
        g_val_neg_list[i].add_edges(edges_0_groups[i][:, 0], edges_0_groups[i][:, 1])  # 测试图neg

    return g_tra_list, g_tra_pos_list, g_tra_neg_list, g_val_pos_list, g_val_neg_list


if __name__ == '__main__':
    dataset = NeuCalDataset()
    # tra_g, tra_pos_g, tra_neg_g, val_pos_g, val_neg_g, tes_pos_g, tes_neg_g = split_train_val_test_with_edge(
    #     dataset.graph, dataset.selected_adj_mat)
    # print(dataset.graph)
    split_edges_with_five_folds(dataset.graph, dataset.selected_adj_mat)

