import sys
import config
import torch
import torch.nn.functional as F
from graph_feat_prepare import NeuCalDataset, split_train_val_test_with_edge, split_edges_with_five_folds
from modules import GraphSAGE, MLPPredictor
from sklearn.metrics import roc_auc_score
import itertools
import numpy as np
import dgl
import random

# 全局变量
best_acc_list = []
best_auc_list = []
best_pre_list = []
best_recall_list = []
best_f1_list = []


# def compute_loss(pos_score, neg_score):
#     scores = torch.cat([pos_score, neg_score])
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
#     )
#     return F.binary_cross_entropy(scores, labels)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    pos_labels = torch.cat([torch.ones(pos_score.shape[0], 1), torch.zeros(pos_score.shape[0], 1)], dim=1)
    neg_labels = torch.cat([torch.zeros(neg_score.shape[0], 1), torch.ones(neg_score.shape[0], 1)], dim=1)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    loss = F.binary_cross_entropy(scores, labels)
    return loss


# def compute_auc(pos_score, neg_score):
#     scores = torch.cat([pos_score, neg_score]).detach().numpy()
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
#     ).numpy()
#     return roc_auc_score(labels, scores)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    pos_labels = torch.cat([torch.ones(pos_score.shape[0], 1), torch.zeros(pos_score.shape[0], 1)], dim=1)
    neg_labels = torch.cat([torch.zeros(neg_score.shape[0], 1), torch.ones(neg_score.shape[0], 1)], dim=1)
    labels = torch.cat([pos_labels, neg_labels], dim=0).numpy()
    auc = roc_auc_score(labels, scores)
    return auc

# def compute_performance(pos_score, neg_score):
#     pos_count = pos_score.shape[0]  # 正样本数量
#     neg_count = neg_score.shape[0]  # 负样本数量
#     ture_pos_count = np.sum(pos_score.detach().numpy() > 0.5)  # 正样本预测正确的数量
#     false_pos_count = np.sum(pos_score.detach().numpy() <= 0.5)  # 正样本预测错误的数量
#     ture_neg_count = np.sum(neg_score.detach().numpy() < 0.5)  # 负样本预测正确的数量
#     false_neg_count = np.sum(neg_score.detach().numpy() >= 0.5)  # 负样本预测错误的数量
#     acc = (ture_pos_count + ture_neg_count) / (pos_count + neg_count)
#     pre = ture_pos_count / (ture_pos_count + false_pos_count)
#     recall = ture_pos_count / (ture_pos_count + false_neg_count + np.finfo(float).tiny)
#     f1 = 2 * pre * recall / (pre + recall + np.finfo(float).tiny)
#     return acc, pre, recall, f1


def compute_performance(pos_score, neg_score):
    pos_count = pos_score.shape[0]  # 正样本数量
    neg_count = neg_score.shape[0]  # 负样本数量
    ture_pos_count = torch.sum(pos_score[:, 0] > pos_score[:, 1]).item()  # 正样本预测正确的数量
    false_pos_count = torch.sum(pos_score[:, 0] < pos_score[:, 1]).item()  # 正样本预测错误的数量
    ture_neg_count = torch.sum(neg_score[:, 0] < neg_score[:, 1]).item()  # 负样本预测正确的数量
    false_neg_count = torch.sum(neg_score[:, 0] > neg_score[:, 1]).item()  # 负样本预测错误的数量
    acc = (ture_pos_count + ture_neg_count) / (pos_count + neg_count)
    pre = ture_pos_count / (ture_pos_count + false_pos_count)
    recall = ture_pos_count / (ture_pos_count + false_neg_count + np.finfo(float).tiny)
    f1 = 2 * pre * recall / (pre + recall + np.finfo(float).tiny)
    return acc, pre, recall, f1


def isolated_nodes_check(g):
    # 检查每个节点是否是孤立点
    nodes_list = g.nodes()
    isolated_nodes = []
    for node in nodes_list:
        if g.in_degrees(node) == 0 and g.out_degrees(node) == 0:
            isolated_nodes.append(node.item())
    if isolated_nodes:
        print("图中存在孤立点，节点列表为:", isolated_nodes)
        print("请重新执行程序，确保不包含孤立点")
        sys.exit()
    else:
        print("图中不存在孤立点")


def train(tra_g, tra_pos_g, tra_neg_g, val_pos_g, val_neg_g, foldsIdx, loopsIdx):

    # 定义模型与优化器
    model = GraphSAGE(tra_g.ndata["feat"].shape[1], config.n_features, config.embedding_dim)
    pred = MLPPredictor(config.embedding_dim)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=config.lr, weight_decay=0.0e-4)

    # 执行训练与验证
    tra_info, val_info = [], []
    best_acc = 0.0
    best_auc = 0.0
    best_pre = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for epoch in range(config.n_epochs):

        model.train()

        # 执行前向迭代
        output = model(tra_g, tra_g.ndata["feat"])
        pos_score, neg_score = pred(tra_pos_g, output), pred(tra_neg_g, output)
        loss = compute_loss(pos_score, neg_score)

        # 执行后向更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练指标
        if epoch % 1 == 0:
            model.eval()
            acc, pre, recall, f1 = compute_performance(pos_score, neg_score)
            auc = compute_auc(pos_score, neg_score)
            print("Epoch: %3d, Loss: %.5f, Auc: %.5f, Acc: %.5f, Pre: %.5f, Recall: %.5f, F1: %.5f"
                  % (epoch, loss.item(), auc, acc, pre, recall, f1))
            tra_info_tmp = [epoch, loss.item(), auc, acc, pre, recall, f1]
            tra_info.append(tra_info_tmp)

        # 计算验证指标
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                pos_score, neg_score = pred(val_pos_g, output), pred(val_neg_g, output)
                acc, pre, recall, f1 = compute_performance(pos_score, neg_score)
                auc, loss = compute_auc(pos_score, neg_score), compute_loss(pos_score, neg_score)
                print("Epoch: %3d, Loss: %.5f, Auc: %.5f, Acc: %.5f, Pre: %.5f, Recall: %.5f, F1: %.5f"
                      % (epoch, loss.item(), auc, acc, pre, recall, f1))
                val_info_tmp = [epoch, loss.item(), auc, acc, pre, recall, f1]
                val_info.append(val_info_tmp)

        # 保存最优数据
        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_pre = pre
            best_recall = recall
            best_f1 = f1
            filename = config.cadata_filepath + '-model-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.pth'
            torch.save(model, filename)
            filename = config.cadata_filepath + '-pred-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.pth'
            torch.save(pred, filename)
            filename = config.cadata_filepath + '-pos_score-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.pth'
            torch.save(pos_score, filename)
            filename = config.cadata_filepath + '-neg_score-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.pth'
            torch.save(neg_score, filename)

    # 将列表转换为numpy数组
    tra_info = np.array(tra_info)
    val_info = np.array(val_info)

    # 保存numpy数组为CSV文件
    filename = config.cadata_filepath + '-tra_info-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.csv'
    np.savetxt(filename, tra_info, delimiter=',', fmt='%s', encoding='utf-8')
    filename = config.cadata_filepath + '-val_info-foldIdx=' + str(foldsIdx) + '-loopsIdx=' + str(loopsIdx) + '.csv'
    np.savetxt(filename, val_info, delimiter=',', fmt='%s', encoding='utf-8')

    print("数据已成功保存！")
    print("best acc %f" % best_acc)
    best_acc_list.append(best_acc)
    best_auc_list.append(best_auc)
    best_pre_list.append(best_pre)
    best_f1_list.append(best_f1)
    best_recall_list.append(best_recall)


def main_v1(index):
    """
    程序读入指定名称的邻接矩阵与钙活性数据，按比例划分训练集、测试集与验证集
    数据集划分比例是固定的，但划分是随机的
    上述划分数据集的方法可能存在问题，解决办法考虑是使用五折交叉或者固定训练集、测试集、验证集
    """

    # ====================================
    # 读入数据，制作数据集
    # ====================================
    dataset = NeuCalDataset()
    tra_g, tra_pos_g, tra_neg_g, val_pos_g, val_neg_g, tes_pos_g, tes_neg_g = (
        split_train_val_test_with_edge(dataset.graph, dataset.selected_adj_mat))
    print('========================================')
    print('tra_g: (node, edge) = (%d, %d)' % (tra_g.number_of_nodes(), tra_g.number_of_edges()))
    print('tra_pos_g: (node, edge) = (%d, %d)' % (tra_pos_g.number_of_nodes(), tra_pos_g.number_of_edges()))
    print('tra_neg_g: (node, edge) = (%d, %d)' % (tra_neg_g.number_of_nodes(), tra_neg_g.number_of_edges()))
    print('tes_pos_g: (node, edge) = (%d, %d)' % (tes_pos_g.number_of_nodes(), tes_pos_g.number_of_edges()))
    print('tes_neg_g: (node, edge) = (%d, %d)' % (tes_neg_g.number_of_nodes(), tes_neg_g.number_of_edges()))
    print('val_pos_g: (node, edge) = (%d, %d)' % (val_pos_g.number_of_nodes(), val_pos_g.number_of_edges()))
    print('val_neg_g: (node, edge) = (%d, %d)' % (val_neg_g.number_of_nodes(), val_neg_g.number_of_edges()))
    print('保存图文件...')
    filename = config.cadata_filepath + '-graphs-' + str(index) + '.bin'
    dgl.save_graphs(filename, [tra_g, tra_pos_g, tra_neg_g, tes_pos_g, tes_neg_g, val_pos_g, val_neg_g])

    # 检查每个节点是否是孤立点
    isolated_nodes_check(tra_g)

    # ====================================
    # 初始化模型，训练模型
    # ====================================
    print('========================================')
    print('Starting training...')
    model = GraphSAGE(tra_g.ndata["feat"].shape[1], config.n_features, config.embedding_dim)
    pred = MLPPredictor(config.embedding_dim)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()),
                                 lr=config.lr, weight_decay=1.0e-3)

    tra_info = []
    val_info = []
    # tes_info = []
    best_acc = 0.0

    for epoch in range(config.n_epochs):
        # 前馈
        output = model(tra_g, tra_g.ndata["feat"])
        pos_score = pred(tra_pos_g, output)
        neg_score = pred(tra_neg_g, output)
        loss = compute_loss(pos_score, neg_score)

        # 后馈
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 指标
        if epoch % 1 == 0:
            acc, pre, recall, f1 = compute_performance(pos_score, neg_score)
            auc = compute_auc(pos_score, neg_score)
            print("Epoch: %3d, Loss: %.5f, Auc: %.5f, Acc: %.5f, Pre: %.5f, Recall: %.5f, F1: %.5f"
                  % (epoch, loss.item(), auc, acc, pre, recall, f1))
            tra_info_tmp = [epoch, loss.item(), auc, acc, pre, recall, f1]
            tra_info.append(tra_info_tmp)

        if epoch % 1 == 0:
            with torch.no_grad():
                pos_score = pred(val_pos_g, output)
                neg_score = pred(val_neg_g, output)
                acc, pre, recall, f1 = compute_performance(pos_score, neg_score)
                auc = compute_auc(pos_score, neg_score)
                loss = compute_loss(pos_score, neg_score)
                print("Epoch: %3d, Loss: %.5f, Auc: %.5f, Acc: %.5f, Pre: %.5f, Recall: %.5f, F1: %.5f"
                      % (epoch, loss.item(), auc, acc, pre, recall, f1))
                val_info_tmp = [epoch, loss.item(), auc, acc, pre, recall, f1]
                val_info.append(val_info_tmp)

        if acc > best_acc:
            best_acc = acc
            filename = config.cadata_filepath + '-model-' + str(index) + '.pth'
            torch.save(model, filename)
            filename = config.cadata_filepath + '-pred-' + str(index) + '.pth'
            torch.save(pred, filename)
            filename = config.cadata_filepath + '-pos_score-' + str(index) + '.pth'
            torch.save(pos_score, filename)
            filename = config.cadata_filepath + '-neg_score-' + str(index) + '.pth'
            torch.save(neg_score, filename)
            # val_adj_mat = gen_val_graph(val_pos_g, val_neg_g, pos_score, neg_score)
            # filename = config.cadata_filepath + '-val_adj_mat.csv'
            # np.savetxt(filename, val_adj_mat, delimiter=',', fmt='%s', encoding='utf-8')

    # 将列表转换为 numpy 数组
    tra_info = np.array(tra_info)
    val_info = np.array(val_info)

    # 保存 numpy 数组为 CSV 文件
    filename = config.cadata_filepath + '-tra_info-' + str(index) + '.csv'
    np.savetxt(filename, tra_info, delimiter=',', fmt='%s', encoding='utf-8')
    filename = config.cadata_filepath + '-val_info-' + str(index) + '.csv'
    np.savetxt(filename, val_info, delimiter=',', fmt='%s', encoding='utf-8')

    print("数据已成功保存！")
    print("best acc %f" % best_acc)


def main_v2(loopsIdx):
    """
    程序读入指定名称的邻接矩阵与钙活性数据，使用五折交叉处理数据集
    """

    # ====================================
    # 读入数据，制作数据集
    # ====================================
    dataset = NeuCalDataset()
    tra_g_ls, tra_pos_g_ls, tra_neg_g_ls, val_pos_g_ls, val_neg_g_ls = split_edges_with_five_folds(dataset.graph, dataset.selected_adj_mat)

    print('========================================')
    print('训练集信息:')
    for idx in range(len(tra_g_ls)):
        print('tra_g: (node, edge) = (%d, %d)' % (tra_g_ls[idx].number_of_nodes(), tra_g_ls[idx].number_of_edges()))
        print('tra_pos_g: (node, edge) = (%d, %d)' % (tra_pos_g_ls[idx].number_of_nodes(), tra_pos_g_ls[idx].number_of_edges()))
        print('tra_neg_g: (node, edge) = (%d, %d)' % (tra_neg_g_ls[idx].number_of_nodes(), tra_neg_g_ls[idx].number_of_edges()))
    print('验证集信息:')
    for idx in range(len(val_pos_g_ls)):
        print('val_pos_g: (node, edge) = (%d, %d)' % (val_pos_g_ls[idx].number_of_nodes(), val_pos_g_ls[idx].number_of_edges()))
        print('val_neg_g: (node, edge) = (%d, %d)' % (val_neg_g_ls[idx].number_of_nodes(), val_neg_g_ls[idx].number_of_edges()))

    print('保存图文件...')
    filename = config.cadata_filepath + '-tra_g-loopsIdx=' + str(loopsIdx) + '.bin'
    dgl.save_graphs(filename, tra_g_ls)
    filename = config.cadata_filepath + '-tra_pos_g-loopsIdx=' + str(loopsIdx) + '.bin'
    dgl.save_graphs(filename, tra_pos_g_ls)
    filename = config.cadata_filepath + '-tra_neg_g-loopsIdx=' + str(loopsIdx) + '.bin'
    dgl.save_graphs(filename, tra_neg_g_ls)
    filename = config.cadata_filepath + '-val_pos_g-loopsIdx=' + str(loopsIdx) + '.bin'
    dgl.save_graphs(filename, val_pos_g_ls)
    filename = config.cadata_filepath + '-val_neg_g-loopsIdx=' + str(loopsIdx) + '.bin'
    dgl.save_graphs(filename, val_neg_g_ls)
    
    print('检查孤立节点:')
    for idx in range(len(tra_g_ls)):
        isolated_nodes_check(tra_g_ls[idx])

    # ====================================
    # 初始化模型，训练模型
    # ====================================
    print('========================================')
    print('Starting training...')
    for foldsIdx in range(len(tra_g_ls)):
        train(tra_g_ls[foldsIdx], tra_pos_g_ls[foldsIdx], tra_neg_g_ls[foldsIdx], 
              val_pos_g_ls[foldsIdx], val_neg_g_ls[foldsIdx], foldsIdx, loopsIdx)
    print('训练结束!')


if __name__ == '__main__':

    # 设置随机数种子
    # seed = random.randint(0, 10000)
    seed = 421
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

    # 运行程序
    main_v2(loopsIdx=0)

    # 均值和数量
    print(config.cadata_filepath)
    avg = sum(best_acc_list) / len(best_acc_list)
    print(best_acc_list)
    print(avg)

    import csv
    with open('LSTM+3GraphSAGE-output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([config.cadata_filepath])
        writer.writerow(['best_acc_list', 'best_auc_list', 'best_pre_list', 'best_f1_list', 'best_recall_list'])
        writer.writerow(best_acc_list)
        writer.writerow(best_auc_list)
        writer.writerow(best_pre_list)
        writer.writerow(best_f1_list)
        writer.writerow(best_recall_list)




