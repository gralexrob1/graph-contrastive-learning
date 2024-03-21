import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv


class DGIInductiveGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(DGIInductiveGConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x = self.layers[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


class DGITransductiveGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(DGITransductiveGConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class GRACEGConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GRACEGConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z
