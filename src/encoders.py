import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import inits

logger = logging.getLogger(__name__)

"""
DGI
"""


class DGIEncoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(DGIEncoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        inits.uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


"""
GRACE
"""


class GRACEEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(GRACEEncoder, self).__init__()

        logger.info("CALL GRACEEncoder")
        logger.info(f"Encoder: {encoder}")
        logger.info(f"Augmentor: {augmentor}")

        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


"""
GraphCL
"""


class GraphCLEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(GraphCLEncoder, self).__init__()

        logger.info("CALL GraphCLEncoder")
        logger.info(f"Encoder: {encoder}")
        logger.info(f"Augmentor: {augmentor}")

        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


"""
InfoGraph
"""


class InfoGraphEncoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(InfoGraphEncoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)
        return self.local_fc(z), self.global_fc(g)
