import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import numpy as np
import seaborn as sns
from sympy.benchmarks.bench_meijerint import alpha

from config import filepath

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def plot_loss_curve(tra_loss, val_loss, linewidth):
    plt.plot(epochs, tra_loss, label='Training Loss', color='blue', linewidth=linewidth)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=linewidth)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, 1000)
    plt.ylim(0, 1.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(250))  # x轴间距为1
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.25))  # y轴间距为2
    plt.legend()


def plot_loss_curve_v2(data_length, tra_loss_mean, tra_loss_std, val_loss_mean, val_loss_std, linewidth):
    epochs = np.arange(data_length)

    upper_bound = tra_loss_mean + tra_loss_std
    lower_bound = tra_loss_mean - tra_loss_std
    plt.plot(epochs, tra_loss_mean, label='Training Loss', color='blue', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='blue', alpha=0.2)  # 填充阴影

    upper_bound = val_loss_mean + val_loss_std
    lower_bound = val_loss_mean - val_loss_std
    plt.plot(epochs, val_loss_mean, label='Validation Loss', color='orange', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='orange', alpha=0.2)  # 填充阴影

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, data_length)
    plt.ylim(0.3, 0.9)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    plt.yticks([0.30, 0.45, 0.60, 0.75, 0.90])
    plt.legend()


def plot_acc_curve(tra_acc, val_acc, linewidth):
    plt.plot(epochs, tra_acc, label='Training ACC', color='blue', linewidth=linewidth)
    plt.plot(epochs, val_acc, label='Validation ACC', color='orange', linewidth=linewidth)
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.xlim(0, 1000)
    plt.ylim(0, 1.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(250))  # x轴间距为1
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.25))  # y轴间距为2
    plt.legend()


def plot_acc_curve_v2(data_length, tra_acc_mean, tra_acc_std, val_acc_mean, val_acc_std, linewidth):
    epochs = np.arange(data_length)

    upper_bound = tra_acc_mean + tra_acc_std
    lower_bound = tra_acc_mean - tra_acc_std
    plt.plot(epochs, tra_acc_mean, label='Training ACC', color='blue', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='blue', alpha=0.2)  # 填充阴影

    upper_bound = val_acc_mean + val_acc_std
    lower_bound = val_acc_mean - val_acc_std
    plt.plot(epochs, val_acc_mean, label='Validation ACC', color='orange', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='orange', alpha=0.2)  # 填充阴影

    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.xlim(0, data_length)
    plt.ylim(0.45, 0.85)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.yticks([0.45, 0.56, 0.67, 0.78, 0.89])
    plt.legend()


def plot_auc_curve(tra_auc, val_auc, linewidth):
    plt.plot(epochs, tra_auc, label='Training AUC', color='blue', linewidth=linewidth)
    plt.plot(epochs, val_auc, label='Validation AUC', color='orange', linewidth=linewidth)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.xlim(0, 1000)
    plt.ylim(0.2, 1.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))  # x轴间距为1
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # y轴间距为2
    plt.legend()


def plot_auc_curve_v2(data_length, tra_auc_mean, tra_auc_std, val_auc_mean, val_auc_std, linewidth):
    epochs = np.arange(data_length)

    upper_bound = tra_auc_mean + tra_auc_std
    lower_bound = tra_auc_mean - tra_auc_std
    plt.plot(epochs, tra_auc_mean, label='Training AUC', color='blue', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='blue', alpha=0.2)  # 填充阴影

    upper_bound = val_auc_mean + val_auc_std
    lower_bound = val_auc_mean - val_auc_std
    plt.plot(epochs, val_auc_mean, label='Validation AUC', color='orange', linewidth=linewidth)  # 绘制均值曲线
    plt.fill_between(epochs, lower_bound, upper_bound, color='orange', alpha=0.2)  # 填充阴影

    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.xlim(0, data_length)
    plt.ylim(0.4, 1.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.yticks([0.40, 0.55, 0.70, 0.85, 1.00])
    plt.legend()


def plot_roc_curve(pos_scores, neg_scores, linewidth):
    # 将 scores 转换为 numpy 数组
    pos_scores = pos_scores.numpy()
    neg_scores = neg_scores.numpy()

    # 创建标签
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, color='blue', label='ROC curve\n(area = {:.2f})'.format(roc_auc), linewidth=linewidth)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=linewidth)  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')


def plot_confusion_matrix(pos_score_path, neg_score_path, threshold=0.5, class_names=None):
    """
    从文件中加载 pos_score 和 neg_score，并基于分数绘制混淆矩阵。

    参数:
    - pos_score_path: 正样本分数文件路径
    - neg_score_path: 负样本分数文件路径
    - threshold: 阈值，用于将分数转换为预测标签 (默认为 0.5)
    - class_names: 类别名称列表，默认为 ['Negative', 'Positive']

    返回:
    - None
    """
    # 从文件加载分数
    if class_names is None:
        class_names = {'Negative', 'Positive'}
    pos_score = torch.load(pos_score_path).numpy()
    neg_score = torch.load(neg_score_path).numpy()

    # 创建真实标签和预测标签
    y_true = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
    y_scores = np.concatenate([pos_score, neg_score])
    y_pred = (y_scores >= threshold).astype(int)  # 使用阈值将分数转换为0或1

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names, cbar=False)

    # 添加标题和轴标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_line_chart_with_markers(data, y_low, y_high):
    indices = np.arange(0.5, 10, 1)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(data)))
    plt.plot(indices, data, linestyle='--', color='k', alpha=0.75, label='Validation ACC')
    for i, value in enumerate(data):
        plt.scatter(indices[i], value, color=colors[i], s=100)
    plt.xlabel('Fold index')
    plt.ylabel('ACC')

    plt.xlim(0, 10)
    plt.ylim(y_low, y_high)
    # plt.yticks([0.65, 0.70, 0.75, 0.80, 0.85])
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=1)  # 设置网格线类型、颜色、粗细


def set_figure():
    fig, axs = plt.subplots(3, 4, figsize=(13, 8))

    # 合并第2行第1、2列为一个子图
    big_ax1 = fig.add_subplot(3, 4, (5, 6))  # 索引 5 和 6
    fig.delaxes(axs[1, 0])  # 删除原来的 axs[1, 0]
    fig.delaxes(axs[1, 1])  # 删除原来的 axs[1, 1]

    # 合并第2行第3、4列为一个子图
    big_ax2 = fig.add_subplot(3, 4, (7, 8))  # 索引 7 和 8
    fig.delaxes(axs[1, 2])  # 删除原来的 axs[1, 2]
    fig.delaxes(axs[1, 3])  # 删除原来的 axs[1, 3]

    # 合并第3行整行为一个子图
    big_ax3 = fig.add_subplot(3, 4, (9, 12))  # 索引 9 到 12
    fig.delaxes(axs[2, 0])  # 删除原来的 axs[2, 0]
    fig.delaxes(axs[2, 1])  # 删除原来的 axs[2, 1]
    fig.delaxes(axs[2, 2])  # 删除原来的 axs[2, 2]
    fig.delaxes(axs[2, 3])  # 删除原来的 axs[2, 3]

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_color('black')  # 设置边框为黑色
            spine.set_linewidth(1)  # 设置边框粗细为2
        ax.tick_params(direction='in', length=6, width=1)  # 全部刻度线向内
        ax.tick_params(axis='x', pad=10)  # x轴刻度距离轴线10个点
        ax.tick_params(axis='y', pad=10)  # y轴刻度距离轴线10个点
        ax.grid(True, color='gray', linestyle='--', linewidth=1)  # 设置网格线类型、颜色、粗细
    return fig


def plot_bar_chart(data, group_labels, y_low=0, y_high=1):
    """
    绘制柱形图的子函数。

    参数：
    - data: 二维列表或数组，形状为 (10, 3)，表示 10 组，每组 3 个数值。
    - group_labels: 横轴的组标签 (list)，长度为 10。
    - y_low: 纵轴的最小值 (默认 0)。
    - y_high: 纵轴的最大值 (默认 1)。
    """
    # 设置柱状图的参数
    num_groups = len(data)  # 组数 (10)
    num_bars_per_group = len(data[0])  # 每组柱子数量 (3)

    # 确定柱子在横轴上的位置
    bar_width = 0.2  # 每个柱子的宽度
    x = np.arange(num_groups)  # 横轴组的索引

    # 颜色映射
    # colors = ['red', 'blue', 'green']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 绘制柱形图
    labels = ['GNN-only(2-layer)', 'LSTM-GNN(1-layer)', 'LSTM-GNN(2-layer)', 'LSTM-GNN(3-layer)']
    for i in range(num_bars_per_group):
        plt.bar(x + i * bar_width, [group[i] for group in data], alpha=1.0,
                width=bar_width, color=colors[i], label=labels[i])

    # 设置轴标签和范围
    plt.xticks(x + bar_width, group_labels)
    plt.ylim(y_low, y_high)
    plt.xlabel('Dataset Index')
    plt.ylabel('ACC')
    plt.legend(ncol=4)
    plt.grid(True, color='gray', linestyle='--', linewidth=1)  # 设置网格线类型、颜色、粗细


if __name__ == '__main__':

    # ========================================================
    # 绘制第一部分数据
    # ========================================================
    # 读入数据
    # tra_loss, val_loss = [], []
    # tra_acc, val_acc = [], []
    # tra_auc, val_auc = [], []
    # data_length = 400
    # filename1 = './Table2实验数据/3rd raw calcium intensity.csv-tra_info-foldIdx='
    # filename2 = './Table2实验数据/3rd raw calcium intensity.csv-val_info-foldIdx='
    # for loopsIdx in range(10):
    #     filepath1 = filename1 + str(loopsIdx) + '-loopsIdx=0.csv'
    #     filepath2 = filename2 + str(loopsIdx) + '-loopsIdx=0.csv'
    #     tra_data = pd.read_csv(filepath1)
    #     val_data = pd.read_csv(filepath2)
    #     # 读取loss数据
    #     tra_loss.append(tra_data.iloc[:data_length, 1])
    #     val_loss.append(val_data.iloc[:data_length, 1])
    #     # 读取acc数据
    #     tra_acc.append(tra_data.iloc[:data_length, 3])
    #     val_acc.append(val_data.iloc[:data_length, 3])

    # tra_loss = np.array(tra_loss)
    # tra_loss_mean = np.mean(tra_loss, axis=0)
    # tra_loss_std = np.std(tra_loss, axis=0)

    # val_loss = np.array(val_loss)
    # val_loss_mean = np.mean(val_loss, axis=0)
    # val_loss_std = np.std(val_loss, axis=0)

    # tra_acc = np.array(tra_acc)
    # tra_acc_mean = np.mean(tra_acc, axis=0)
    # tra_acc_std = np.std(tra_acc, axis=0)

    # val_acc = np.array(val_acc)
    # val_acc_mean = np.mean(val_acc, axis=0)
    # val_acc_std = np.std(val_acc, axis=0)

    # val_acc_max = np.max(val_acc, axis=1)

    # # val_acc_max_mean = np.mean(val_acc_max, axis=0)
    # print(np.max(val_acc_max))
    # sys.exit(0)

    # 创建图片设置属性
    fig = set_figure()

    # 绘制曲线
    plt.subplot(3, 4, 1)
    # plot_loss_curve_v2(data_length, tra_loss_mean, tra_loss_std, val_loss_mean, val_loss_std, 2)
    plt.subplot(3, 4, 2)
    # plot_acc_curve_v2(data_length, tra_acc_mean, tra_acc_std, val_acc_mean, val_acc_std, 2)
    # plt.subplot2grid((3, 4), (1, 0), colspan=2)
    plt.subplot(3, 4, (5, 6))
    # data = val_acc_max
    # plot_line_chart_with_markers(data, 0.60, 0.85)

    # plt.show()
    # sys.exit(0)

    # ========================================================
    # 绘制第二部分数据
    # ========================================================
    # 读入数据
    # tra_loss, val_loss = [], []
    # tra_acc, val_acc = [], []
    # tra_auc, val_auc = [], []
    # data_length = 500
    # filename1 = './消融实验2单层GCN/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv-tra_info-foldIdx='
    # filename2 = './消融实验2单层GCN/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv-val_info-foldIdx='
    # for loopsIdx in range(10):
    #     filepath1 = filename1 + str(loopsIdx) + '-loopsIdx=0.csv'
    #     filepath2 = filename2 + str(loopsIdx) + '-loopsIdx=0.csv'
    #     tra_data = pd.read_csv(filepath1)
    #     val_data = pd.read_csv(filepath2)
    #     # 读取loss数据
    #     tra_loss.append(tra_data.iloc[:data_length, 1])
    #     val_loss.append(val_data.iloc[:data_length, 1])
    #     # 读取acc数据
    #     tra_acc.append(tra_data.iloc[:data_length, 3])
    #     val_acc.append(val_data.iloc[:data_length, 3])

    # tra_loss = np.array(tra_loss)
    # tra_loss_mean = np.mean(tra_loss, axis=0)
    # tra_loss_std = np.std(tra_loss, axis=0)

    # val_loss = np.array(val_loss)
    # val_loss_mean = np.mean(val_loss, axis=0)
    # val_loss_std = np.std(val_loss, axis=0)

    # tra_acc = np.array(tra_acc)
    # tra_acc_mean = np.mean(tra_acc, axis=0)
    # tra_acc_std = np.std(tra_acc, axis=0)

    # val_acc = np.array(val_acc)
    # val_acc_mean = np.mean(val_acc, axis=0)
    # val_acc_std = np.std(val_acc, axis=0)

    # val_acc_max = np.max(val_acc, axis=1)
    # val_acc_max_mean = np.mean(val_acc_max, axis=0)

    # # 绘制曲线
    # plt.subplot(3, 4, 3)
    # plot_loss_curve_v2(data_length, tra_loss_mean, tra_loss_std, val_loss_mean, val_loss_std, 2)
    # plt.subplot(3, 4, 4)
    # plot_acc_curve_v2(data_length, tra_acc_mean, tra_acc_std, val_acc_mean, val_acc_std, 2)
    # plt.subplot(3, 4, (7, 8))
    # # plt.subplot2grid((3, 4), (1, 2), colspan=2)
    # data = val_acc_max
    # plot_line_chart_with_markers(data, 0.60, 0.85)

    # ========================================================
    # 绘制第三部分数据
    # ========================================================
    data = [
        [0.7321, 0.7179, 0.7462, 0.7449], 
        [0.6858, 0.7070, 0.7369, 0.7060], 
        [0.6557, 0.6908, 0.6887, 0.6732], 
        [0.6711, 0.6938, 0.7018, 0.6923], 
        [0.6742, 0.6895, 0.7373, 0.7057], 
        [0.6936, 0.7205, 0.7113, 0.6776], 
        [0.6844, 0.6961, 0.7041, 0.6828], 
        [0.6928, 0.6877, 0.7271, 0.7453], 
        [0.6900, 0.6968, 0.7252, 0.7014], 
        [0.7172, 0.6994, 0.7283, 0.7196]
    ]
    plt.subplot(3, 4, (9, 12))
    # plt.subplot2grid((3, 4), (2, 0), colspan=4)
    group_labels = [str(i + 1) for i in range(10)]
    plot_bar_chart(data, group_labels, 0.65, 0.80)
    fig.tight_layout()
    plt.show()


