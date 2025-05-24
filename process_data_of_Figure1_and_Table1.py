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
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(250))   # x轴间距为1
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
    plt.yticks([0.45, 0.55, 0.65, 0.75, 0.85])
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


def set_figure():
    fig, axs = plt.subplots(2, 4, figsize=(13, 8/3*2))
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_color('black')  # 设置边框为黑色
            spine.set_linewidth(1)  # 设置边框粗细为2
        ax.tick_params(direction='in', length=6, width=1)  # 全部刻度线向内
        ax.tick_params(axis='x', pad=10)  # x轴刻度距离轴线10个点
        ax.tick_params(axis='y', pad=10)  # y轴刻度距离轴线10个点
        ax.grid(True, color='gray', linestyle='--', linewidth=1)  # 设置网格线类型、颜色、粗细
    # fig.axes[2].set_visible(False)
    # fig.axes[3].set_visible(False)
    # fig.axes[6].set_visible(False)
    # fig.axes[7].set_visible(False)
    return fig


if __name__ == '__main__':

    # # ========================================================
    # # 绘制第一部分数据
    # # ========================================================
    # # 读入数据
    # tra_loss, val_loss = [], []
    # tra_acc, val_acc = [], []
    # tra_auc, val_auc = [], []
    # data_length = 500
    # filename1 = './data/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv-tra_info-foldIdx='
    # filename2 = './data/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv-val_info-foldIdx='
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
    #     # 读取auc数据
    #     tra_auc.append(tra_data.iloc[:data_length, 2])
    #     val_auc.append(val_data.iloc[:data_length, 2])
    #
    # tra_loss = np.array(tra_loss)
    # tra_loss_mean = np.mean(tra_loss, axis=0)
    # tra_loss_std = np.std(tra_loss, axis=0)
    #
    # val_loss = np.array(val_loss)
    # val_loss_mean = np.mean(val_loss, axis=0)
    # val_loss_std = np.std(val_loss, axis=0)
    #
    # tra_acc = np.array(tra_acc)
    # tra_acc_mean = np.mean(tra_acc, axis=0)
    # tra_acc_std = np.std(tra_acc, axis=0)
    #
    # val_acc = np.array(val_acc)
    # val_acc_mean = np.mean(val_acc, axis=0)
    # val_acc_std = np.std(val_acc, axis=0)
    #
    # val_acc_max = np.max(val_acc, axis=1)
    # val_acc_max_mean = np.mean(val_acc_max, axis=0)
    # # print('acc: %f' % val_acc_max_mean)
    # # sys.exit(0)
    #
    # tra_auc = np.array(tra_auc)
    # tra_auc_mean = np.mean(tra_auc, axis=0)
    # tra_auc_std = np.std(tra_auc, axis=0)
    #
    # val_auc = np.array(val_auc)
    # val_auc_mean = np.mean(val_auc, axis=0)
    # val_auc_std = np.std(val_auc, axis=0)
    #
    # # 创建图片设置属性
    # fig = set_figure()
    #
    # # 绘制曲线
    # plt.subplot(2, 4, 1)
    # plot_loss_curve_v2(data_length, tra_loss_mean, tra_loss_std, val_loss_mean, val_loss_std, 2)
    # plt.subplot(2, 4, 2)
    # plot_acc_curve_v2(data_length, tra_acc_mean, tra_acc_std, val_acc_mean, val_acc_std, 2)
    # plt.subplot(2, 4, 3)
    # data = val_acc_max
    # plot_line_chart_with_markers(data, 0.65, 0.85)
    # plt.subplot(2, 4, 4)
    # plot_auc_curve_v2(data_length, tra_auc_mean, tra_auc_std, val_auc_mean, val_auc_std, 2)
    # # plt.show()
    # # sys.exit(0)
    #
    # # ========================================================
    # # 绘制第二部分数据
    # # ========================================================
    # # 读入数据
    # tra_loss, val_loss = [], []
    # tra_acc, val_acc = [], []
    # tra_auc, val_auc = [], []
    # data_length = 500
    # filename1 = './data/20200720LZ_oh16230-D1-10MIN-1003Z-Corr_project_bleach_correction.csv-tra_info-foldIdx='
    # filename2 = './data/20200720LZ_oh16230-D1-10MIN-1003Z-Corr_project_bleach_correction.csv-val_info-foldIdx='
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
    #     # 读取auc数据
    #     tra_auc.append(tra_data.iloc[:data_length, 2])
    #     val_auc.append(val_data.iloc[:data_length, 2])
    #
    # tra_loss = np.array(tra_loss)
    # tra_loss_mean = np.mean(tra_loss, axis=0)
    # tra_loss_std = np.std(tra_loss, axis=0)
    #
    # val_loss = np.array(val_loss)
    # val_loss_mean = np.mean(val_loss, axis=0)
    # val_loss_std = np.std(val_loss, axis=0)
    #
    # tra_acc = np.array(tra_acc)
    # tra_acc_mean = np.mean(tra_acc, axis=0)
    # tra_acc_std = np.std(tra_acc, axis=0)
    #
    # val_acc = np.array(val_acc)
    # val_acc_mean = np.mean(val_acc, axis=0)
    # val_acc_std = np.std(val_acc, axis=0)
    #
    # val_acc_max = np.max(val_acc, axis=1)
    # val_acc_max_mean = np.mean(val_acc_max, axis=0)
    # print('acc: %f' % val_acc_max_mean)
    #
    # tra_auc = np.array(tra_auc)
    # tra_auc_mean = np.mean(tra_auc, axis=0)
    # tra_auc_std = np.std(tra_auc, axis=0)
    #
    # val_auc = np.array(val_auc)
    # val_auc_mean = np.mean(val_auc, axis=0)
    # val_auc_std = np.std(val_auc, axis=0)
    #
    # # 创建图片设置属性
    # # fig = set_figure()
    #
    # # 绘制曲线
    # plt.subplot(2, 4, 5)
    # plot_loss_curve_v2(data_length, tra_loss_mean, tra_loss_std, val_loss_mean, val_loss_std, 2)
    # plt.subplot(2, 4, 6)
    # plot_acc_curve_v2(data_length, tra_acc_mean, tra_acc_std, val_acc_mean, val_acc_std, 2)
    # plt.subplot(2, 4, 7)
    # data = val_acc_max
    # plot_line_chart_with_markers(data, 0.55, 0.75)
    # plt.subplot(2, 4, 8)
    # plot_auc_curve_v2(data_length, tra_auc_mean, tra_auc_std, val_auc_mean, val_auc_std, 2)
    #
    # fig.tight_layout()
    # plt.show()


    # ========================================================
    # 按照acc取最大值找到对应的acc\auc\pre\f1\recall
    # ========================================================
    filename_ls = [
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1003Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1005Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1007Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1009Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1011Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1013Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1015-1Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1015-2Z-Corr_project_bleach_correction.csv-val_info-foldIdx=',
        './消融实验2单层GraphSAGE/20200720LZ_oh16230-D1-10MIN-1017Z-Corr_project_bleach_correction.csv-val_info-foldIdx='
    ]
    folder_num, data_len, data_cls = 10, 500, 7
    for filename in filename_ls:
        val_data = np.zeros((folder_num, data_cls))
        for loopsIdx in range(folder_num):
            filepath = filename + str(loopsIdx) + '-loopsIdx=0.csv'
            val_data_tmp = pd.read_csv(filepath)
            val_data_tmp = np.array(val_data_tmp)[:data_len, :]

            col_index = 3
            row_index = np.argmax(val_data_tmp[:, col_index])
            max_row = val_data_tmp[row_index, :]
            val_data[loopsIdx, :] = max_row
        col_means = np.mean(val_data, axis=0)
        import csv
        with open('output.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(col_means)




























    # # 读入数据
    # filepath = './第2次实验结果/20200720LZ_oh16230-D1-10MIN-1001Z-Corr_project_bleach_correction.csv'
    # tra_data = pd.read_csv(filepath + '-tra_info.csv')
    # val_data = pd.read_csv(filepath + '-val_info.csv')
    # epochs = tra_data.iloc[:1000, 0]  # 第一列epoch数据
    #
    # # 绘制loss曲线
    # tra_loss = tra_data.iloc[:1000, 1]  # 第二列train loss数据
    # val_loss = val_data.iloc[:1000, 1]  # 第二列val loss数据
    # plt.subplot(3, 4, 1)
    # plot_loss_curve(tra_loss, val_loss, linewidth=1)
    #
    # # 绘制acc曲线
    # tra_acc = tra_data.iloc[:1000, 3]  # 第二列train acc数据
    # val_acc = val_data.iloc[:1000, 3]  # 第二列val acc数据
    # plt.subplot(3, 4, 2)
    # plot_acc_curve(tra_acc, val_acc, linewidth=1)
    #
    # # 绘制auc曲线
    # tra_auc = tra_data.iloc[:1000, 2]  # 第二列train auc数据
    # val_auc = val_data.iloc[:1000, 2]  # 第二列val auc数据
    # plt.subplot(3, 4, 5)
    # plot_auc_curve(tra_auc, val_auc, linewidth=1)
    #
    # # 绘制roc曲线
    # pos_scores = torch.load(filepath + '-pos_score.pth')
    # neg_scores = torch.load(filepath + '-neg_score.pth')
    # plt.subplot(3, 4, 6)
    # plot_roc_curve(pos_scores, neg_scores, linewidth=1)
    #
    # # 读入数据
    # filepath = './第2次实验结果/20200720LZ_oh16230-D1-10MIN-1003Z-Corr_project_bleach_correction.csv'
    # tra_data = pd.read_csv(filepath + '-tra_info.csv')
    # val_data = pd.read_csv(filepath + '-val_info.csv')
    # epochs = tra_data.iloc[:1000, 0]  # 第一列epoch数据
    #
    # # 绘制loss曲线
    # tra_loss = tra_data.iloc[:1000, 1]  # 第二列train loss数据
    # val_loss = val_data.iloc[:1000, 1]  # 第二列val loss数据
    # plt.subplot(3, 4, 9)
    # plot_loss_curve(tra_loss, val_loss, linewidth=1)
    #
    # # 绘制acc曲线
    # tra_acc = tra_data.iloc[:1000, 3]  # 第二列train acc数据
    # val_acc = val_data.iloc[:1000, 3]  # 第二列val acc数据
    # plt.subplot(3, 4, 10)
    # plot_acc_curve(tra_acc, val_acc, linewidth=1)
    #
    # # 绘制auc曲线
    # tra_auc = tra_data.iloc[:1000, 2]  # 第二列train auc数据
    # val_auc = val_data.iloc[:1000, 2]  # 第二列val auc数据
    # plt.subplot(3, 4, 11)
    # plot_auc_curve(tra_auc, val_auc, linewidth=1)
    #
    # # 绘制roc曲线
    # pos_scores = torch.load(filepath + '-pos_score.pth')
    # neg_scores = torch.load(filepath + '-neg_score.pth')
    # plt.subplot(3, 4, 12)
    # plot_roc_curve(pos_scores, neg_scores, linewidth=1)
    #
    # fig.tight_layout()
    # plt.show()
    #
    # # 绘制混淆矩阵
    # plot_confusion_matrix(filepath + '-pos_score.pth', filepath + '-neg_score.pth', threshold=0.5)


