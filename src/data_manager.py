import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data


def generate_ogbn_proteins_dataset():
    proteins_dataset = PygNodePropPredDataset(name="ogbn-proteins")

    split_idx = proteins_dataset.get_idx_split()
    _, _, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph = proteins_dataset[0]
    train_graph = graph.subgraph(test_idx)

    node_embedding = torch.load("../ogb-proteins-node2vec/embedding.pt")
    train_node_embeddings = node_embedding[test_idx]

    enriched_train_graph = Data(
        num_nodes=train_graph.num_nodes,
        edge_index=train_graph.edge_index,
        edge_attr=train_graph.edge_attr,
        node_species=train_graph.node_species,
        x=train_node_embeddings,
        y=train_graph.y,
    )

    return enriched_train_graph
