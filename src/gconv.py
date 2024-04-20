import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool


def make_gin_conv(input_dim, out_dim):
    return GINConv(
        nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    )


"""
DGI Inductive
"""


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


"""
DGI Transductive
"""


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


"""
GRACE
"""


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


"""
GraphCL
"""


class GraphCLGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GraphCLGConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim),
        )

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


"""
InfoGraph
"""


class InfoGraphGConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(InfoGraphGConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)
