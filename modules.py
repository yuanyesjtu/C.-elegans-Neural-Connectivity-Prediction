import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv, GATConv, GatedGraphConv


class Encoder(nn.Module):
    """
    定义一个编码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        # 使用双层LSTM
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True)

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return x


class Decoder(nn.Module):
    """
    定义一个解码器的子类，继承父类 nn.Modul
    """

    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True)

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x = self.output_layer(x)
        return x


class LSTM(nn.Module):
    """
    定义一个自动编码器的子类，继承父类 nn.Module
    并且自动编码器通过编码器和解码器传递输入
    """

    def __init__(self, seq_len, n_features, embedding_dim):
        super(LSTM, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, dropout_prob=0.5):
        super(GraphSAGE, self).__init__()
        self.seq_len, self.n_features, self.embedding_dim = seq_len, n_features, embedding_dim
        self.in_feats = self.seq_len * self.embedding_dim
        # self.in_feats = self.seq_len * self.n_features
        self.out_feats = self.embedding_dim
        self.encoder = Encoder(self.seq_len, self.n_features, self.embedding_dim)
        self.conv1 = SAGEConv(self.in_feats, self.out_feats, "gcn")
        self.conv2 = SAGEConv(self.out_feats, self.out_feats, "gcn")
        self.conv3 = SAGEConv(self.out_feats, self.out_feats, "gcn")
        # self.conv1 = GraphConv(self.in_feats, self.out_feats, allow_zero_in_degree=True)
        # self.conv1 = GATConv(self.in_feats, self.out_feats, num_heads=1, allow_zero_in_degree=True)
        # self.conv1 = GatedGraphConv(self.in_feats, self.out_feats, n_steps=1, n_etypes=1)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, g, in_feat):
        # in_feat: (num_node, seq_len) -> (num_node, seq_len, n_features)
        in_feat = in_feat.unsqueeze(2)
        in_feat = self.encoder(in_feat)
        in_feat = in_feat.reshape(in_feat.shape[0], -1)
        h = self.conv1(g, in_feat)
        # h = h.squeeze(1)
        h = self.conv2(g, h)
        h = self.conv3(g, h)
        o = F.relu(h)
        return o


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, h_feats)
        self.W3 = nn.Linear(h_feats, 2)

    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        h = F.relu(self.W1(h))
        h = F.relu(self.W2(h))
        score = F.sigmoid(self.W3(h)).squeeze(1)
        return {"score": score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

