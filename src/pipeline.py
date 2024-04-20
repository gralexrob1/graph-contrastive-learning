import logging

import GCL.augmentors as A
import GCL.losses as L
import GCL.models as M
import torch
from GCL.eval import LREvaluator, SVMEvaluator, get_split
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from encoders import DGIEncoder, GRACEEncoder, GraphCLEncoder, InfoGraphEncoder
from gconv import (
    FC,
    DGIInductiveGConv,
    DGITransductiveGConv,
    GRACEGConv,
    GraphCLGConv,
    InfoGraphGConv,
)

logger = logging.getLogger(__name__)


class GCLPipeline:
    def __init__(self, method, contrast_model, augmentations, negative):
        self.method = method
        self.augmentations = augmentations
        self.negative = negative
        self.contrast_model = contrast_model

    @staticmethod
    def init_dataset(dataset_name, data_path, transform=None, batch_size=False):

        logger.info("CALL GCLPipeline.init_dataset")

        match dataset_name:

            case "Cora":
                dataset = Planetoid(data_path, dataset_name, transform=transform)
                num_features = dataset.num_features
                if batch_size > 0:
                    pass
                else:
                    dataset = dataset[0]

            case "PTC-MR" | "PTC_MR":
                dataset = TUDataset(data_path, "PTC_MR")
                num_features = dataset.num_features
                if batch_size > 0:
                    dataset = DataLoader(dataset, batch_size=batch_size)

        logger.info(f"\t Number of features: {num_features}")

        return dataset, num_features

    @staticmethod
    def init_objective(objective_name):
        match objective_name:
            case "InfoNCE":
                return L.InfoNCE(tau=0.2)
            case "JSD":
                return L.JSD()
            case "Triplet":
                return L.TripletMargin()
            case "BootstrapLatent":
                return L.BootstrapLatent()
            case "BarlowTwins":
                return L.BarlowTwins()
            case "VICReg":
                return L.VICReg()
            case _:
                raise NameError(f"Unknown objective name: {objective_name}")

    @staticmethod
    def init_contrast_model(architecture_name, objective, mode):
        match architecture_name:
            case "SingleBranch":
                return M.SingleBranchContrast(objective, mode)
            case "DualBranch":
                return M.DualBranchContrast(objective, mode)
            case "BootstrapBranch":
                return M.BootstrapContrast(objective, mode)
            case "WithinEmbed":
                return M.WithinEmbedContrast(objective)
            case _:
                raise NameError(f"Unknown strategy name: {architecture_name}")

    @staticmethod
    def init_augmentation(name, params=None):
        logger.info("CALL GCLPipeline.init_augmentation")
        logger.info(f"\t Name: {name}")
        match name:
            case "EdgeAdding":
                return A.EdgeAdding() if params is None else A.EdgeAdding(**params)
            case "EdgeAttrMasking":
                return A.EdgeAttrMasking() if params is None else A.EdgeAttrMasking(**params)
            case "EdgeRemoving":
                return A.EdgeRemoving() if params is None else A.EdgeRemoving(**params)
            case "EgoNet":
                return A.Identity() if params is None else A.Identity(**params)
            case "FeatureDropout":
                return A.FeatureDropout() if params is None else A.FeatureDropout(**params)
            case "FeatureMasking":
                return A.FeatureMasking() if params is None else A.FeatureMasking(**params)
            case "Identity":
                return A.Identity() if params is None else A.Identity(**params)
            case "MDK":
                return A.MarkovDiffusion() if params is None else A.MarkovDiffusion(**params)
            case "NodeDropping":
                return A.NodeDropping() if params is None else A.NodeDropping(**params)
            case "NodeShuffling":
                return A.NodeShuffling() if params is None else A.NodeShuffling(**params)
            case "PPRDiffusion":
                return A.PPRDiffusion() if params is None else A.PPRDiffusion(**params)
            case "RWSampling":
                return A.RWSampling() if params is None else A.RWSampling(**params)
            case _:
                raise NotImplementedError(f"Unknown augmentation name: {name}")

    @staticmethod
    def init_augmentations(augmentation, strategy):

        logger.info("CALL GCLPipeline.init_augmentations")
        logger.info(f"\t Strategy: {strategy}")

        if isinstance(augmentation, list):
            augmentations = []
            for a in augmentation:
                augmentations.append(GCLPipeline.init_augmentation(a["name"], a["params"]))
            match strategy:
                case "Random":
                    return A.RandomChoice(augmentations, 1)
                case "Compose":
                    return A.Compose(augmentations)
        else:
            return GCLPipeline.init_augmentation(augmentation["name"], augmentation["params"])

    @classmethod
    def from_strategy(cls, strategy, device):

        method_name = strategy["method"]

        logger.info(f"##### {method_name} #####")

        architecture_name = strategy["architecture"]
        mode_name = strategy["mode"]
        negative_name = strategy["negative"]
        objective_name = strategy["objective"]

        augmentations1 = strategy["augmentation1"]
        augmentations1_strategy = strategy["augmentation1_strat"]
        augmentations2 = strategy["augmentation2"]
        augmentations2_strategy = strategy["augmentation2_strat"]

        logger.info(f"\t Augmentation strategy 1: {augmentations1_strategy}")
        logger.info(f"\t Augmentation strategy 2: {augmentations2_strategy}")

        assert not (architecture_name == "SingleBranch" and mode_name != "G2L")
        assert not (
            architecture_name in ["DualBranch", "Bootstrap"]
            and mode_name not in ["L2L", "G2G", "G2L"]
        )
        assert not (architecture_name == "WithinEmbedding" and mode_name not in ["L2L", "G2G"])

        objective = GCLPipeline.init_objective(objective_name)
        contrast_model = GCLPipeline.init_contrast_model(
            architecture_name,
            objective,
            mode_name,
        ).to(device)

        augmentations = [
            (
                GCLPipeline.init_augmentations(augmentations1, augmentations1_strategy)
                if augmentations1 is not None
                else None
            ),
            (
                GCLPipeline.init_augmentations(augmentations2, augmentations2_strategy)
                if augmentations2 is not None
                else None
            ),
        ]
        logging.info(f"Augmentations: {augmentations}")

        instance = cls(method_name, contrast_model, augmentations, negative_name)

        return instance

    def init_encoder(self, params, device):

        logger.info("CALL GCLPipeline.init_encoder")

        input_dim = params["input_dim"]
        hidden_dim = params["hidden_dim"]
        num_layers = params["num_layers"]
        proj_dim = params["proj_dim"]

        if params["activation"] is None:
            activation = None
        else:
            activation = getattr(
                torch.nn,
                params["activation"],
                ValueError(f"Activation function '{params['activation']}' not found in torch.nn"),
            )

        logger.info(f"\t Input dimension: {input_dim}")
        logger.info(f"\t Hidden dimension: {hidden_dim}")
        logger.info(f"\t Number of layers: {num_layers}")
        logger.info(f"\t Projection dimension: {proj_dim}")
        logger.info(f"\t Activation: {activation}")

        augmentor1 = self.augmentations[0]
        augmentor2 = self.augmentations[1]

        match self.method:

            case "InductiveDGI":
                gconv = DGIInductiveGConv(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
                ).to(device)
                encoder_model = DGIEncoder(encoder=gconv, hidden_dim=hidden_dim).to(device)

            case "TransductiveDGI":
                gconv = DGITransductiveGConv(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
                ).to(device)
                encoder_model = DGIEncoder(encoder=gconv, hidden_dim=hidden_dim).to(device)

            case "GRACE":
                gconv = GRACEGConv(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    num_layers=num_layers,
                ).to(device)
                encoder_model = GRACEEncoder(
                    encoder=gconv,
                    augmentor=(augmentor1, augmentor2),
                    hidden_dim=hidden_dim,
                    proj_dim=proj_dim,
                ).to(device)

            case "GraphCL":
                gconv = GraphCLGConv(
                    input_dim=input_dim,
                    hidden_dim=32,
                    num_layers=2,
                ).to(device)
                encoder_model = GraphCLEncoder(
                    encoder=gconv,
                    augmentor=(augmentor1, augmentor2),
                ).to(device)

            case "InfoGraph":
                gconv = InfoGraphGConv(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    num_layers=num_layers,
                ).to(device)
                fc1 = FC(hidden_dim=hidden_dim * 2)
                fc2 = FC(hidden_dim=hidden_dim * 2)
                encoder_model = InfoGraphEncoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(
                    device
                )

            case _:
                raise NotImplementedError

        return encoder_model

    def train_epoch(self, encoder_model, dataset, optimizer, device):
        """
        Train for 1 epoch

        dataset parameter can be a dataset or a dataloader
        """

        encoder_model.train()

        match self.method:

            case "TransductiveDGI":
                optimizer.zero_grad()
                z, g, zn = encoder_model(dataset.x.to(device), dataset.edge_index.to(device))
                loss = self.contrast_model(h=z, g=g, hn=zn)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            case "GRACE":
                optimizer.zero_grad()
                _, z1, z2 = encoder_model(
                    dataset.x.to(device),
                    dataset.edge_index.to(device),
                    (dataset.edge_attr.to(device) if dataset.edge_attr is not None else None),
                )
                h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
                loss = self.contrast_model(h1, h2)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            case "GraphCL":
                epoch_loss = 0
                for data in dataset:
                    data = data.to(device)
                    optimizer.zero_grad()

                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones(
                            (num_nodes, 1),
                            dtype=torch.float32,
                            device=data.batch.device,
                        )

                    _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
                    g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
                    loss = self.contrast_model(g1=g1, g2=g2, batch=data.batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            case "InfoGraph":
                epoch_loss = 0
                for data in dataset:
                    data = data.to(device)
                    optimizer.zero_grad()

                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones(
                            (num_nodes, 1),
                            dtype=torch.float32,
                            device=data.batch.device,
                        )

                    z, g = encoder_model(data.x, data.edge_index, data.batch)
                    z, g = encoder_model.project(z, g)
                    loss = self.contrast_model(h=z, g=g, batch=data.batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

        return epoch_loss

    def test(self, encoder_model, dataset, device):

        encoder_model.eval()

        match self.method:

            case "TransductiveDGI":
                z, _, _ = encoder_model(dataset.x.to(device), dataset.edge_index.to(device))
                split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
                result = LREvaluator()(z, dataset.y, split)

            case "GRACE":
                z, _, _ = encoder_model(
                    dataset.x.to(device),
                    dataset.edge_index.to(device),
                    (dataset.edge_attr.to(device) if dataset.edge_attr is not None else None),
                )
                split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
                result = LREvaluator()(z, dataset.y, split)

            case "GraphCL":
                x = []
                y = []
                for data in dataset:
                    data = data.to(device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones(
                            (num_nodes, 1),
                            dtype=torch.float32,
                            device=data.batch.device,
                        )
                    _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
                    x.append(g)
                    y.append(data.y)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)

                split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
                result = SVMEvaluator(linear=True)(x, y, split)

            case "InfoGraph":
                x = []
                y = []
                for data in dataset:
                    data = data.to(device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones(
                            (num_nodes, 1),
                            dtype=torch.float32,
                            device=data.batch.device,
                        )
                    z, g = encoder_model(data.x, data.edge_index, data.batch)
                    x.append(g)
                    y.append(data.y)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)

                split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
                result = SVMEvaluator(linear=True)(x, y, split)

        return result
        return result
        return result
