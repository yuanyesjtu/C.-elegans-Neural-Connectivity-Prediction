# 数据集的参数设置
filepath = './ca_data.csv'
tra_length = 0.75
tes_length = 0.15
val_length = 0.10
batch_size = 1
shuffle = True
filter_size = 15

# lstm的参数设置
input_size = 1
embedding_dim = 32
seq_length = 293

# gnn的参数设置
in_feats = seq_length * embedding_dim


# 训练参数
lr = 0.0001
n_epochs = 501


# GNN参数
adjmat_filepath = './data/adj_mat.csv'
cadata_filepath = './data/20200720LZ_oh16230-D1-10MIN-1017Z-Corr_project_bleach_correction.csv'
val_ratio = 0.10
tes_ratio = 0.00
n_features = 1
