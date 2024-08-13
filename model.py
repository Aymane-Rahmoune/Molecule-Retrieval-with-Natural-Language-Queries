import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as gnn
from transformers import AutoModel


class RGNNConv(nn.Module):  # RGAT-LSTM from https://arxiv.org/pdf/1904.08035.pdf
    def __init__(self, in_channels, out_channels, heads=1, dropout=0):
        super(RGNNConv, self).__init__()
        nhid = out_channels * heads
        self.conv = gnn.GATv2Conv(
            in_channels, out_channels, heads=heads, dropout=dropout
        )
        self.ln = nn.LayerNorm(nhid)
        self.rnn = nn.LSTM(nhid, nhid, batch_first=True)

    def forward(self, x, edge_index, h):
        x = self.conv(x, edge_index)
        x = self.ln(x)
        x = F.elu(x)
        x, h = self.rnn(x.unsqueeze(1), h)
        return x.squeeze(1), h


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        nheads=4,
        dropout_rate=0.2,
    ):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.rnn_hid = graph_hidden_channels * nheads
        self.lin0 = nn.Linear(num_node_features, self.rnn_hid)
        self.rnn0 = nn.LSTM(self.rnn_hid, self.rnn_hid, batch_first=True)
        self.conv1 = RGNNConv(num_node_features, graph_hidden_channels, heads=nheads)
        self.conv2 = RGNNConv(
            graph_hidden_channels * nheads,
            graph_hidden_channels,
            heads=nheads,
            dropout=dropout_rate,
        )
        self.conv3 = RGNNConv(
            graph_hidden_channels * nheads,
            graph_hidden_channels,
            heads=nheads,
            dropout=dropout_rate,
        )
        self.mol_hidden1 = nn.Linear(graph_hidden_channels * nheads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
        self.ln1 = nn.LayerNorm(nhid)
        self.ln2 = nn.LayerNorm(nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        x_initial_state = self.lin0(x)
        h0 = torch.zeros(1, x.size(0), self.rnn_hid, device=x.device)
        c0 = torch.zeros(1, x.size(0), self.rnn_hid, device=x.device)
        _, h = self.rnn0(x_initial_state.unsqueeze(1), (h0, c0))

        x, h = self.conv1(x, edge_index, h)
        x, h = self.conv2(x, edge_index, h)
        x, h = self.conv3(x, edge_index, h)

        x = gnn.global_mean_pool(x, batch)
        x = self.mol_hidden1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.mol_hidden2(x)
        x = self.ln2(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name, nout):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        nhid = self.bert.config.hidden_size
        self.proj = nn.Linear(nhid, nout)
        self.ln = nn.LayerNorm(nout)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = encoded_text.last_hidden_state[:, 0, :]
        embeddings = self.proj(embeddings)
        embeddings = self.ln(embeddings)
        return embeddings


class Model(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, graph_hidden_channels
    ):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(
            num_node_features, nout, nhid, graph_hidden_channels
        )
        self.text_encoder = TextEncoder(model_name, nout)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
