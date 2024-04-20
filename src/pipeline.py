import GCL.augmentors as A
import GCL.losses as L
import GCL.models as M
import torch
from GCL.eval import LREvaluator, SVMEvaluator, get_split
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from encoders import DGIEncoder, GRACEEncoder, InfoGraphEncoder
from gconv import (
    FC,
    DGIInductiveGConv,
    DGITransductiveGConv,
    GRACEGConv,
    InfoGraphGConv,
)


class GCLPipeline:
    def __init__(self, method, contrast_model, augmentations, negative):
        self.method = method
        self.augmentations = augmentations
        self.negative = negative
        self.contrast_model = contrast_model

    @staticmethod
    def init_dataset(dataset_name, data_path, transform=None, batch_size=False):

        print("Dataset initialization")

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

        print(f"\t # features: {num_features}")

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
    def init_augmentation(augmentation_name):
        match augmentation_name:
            case "EdgeAdding":
                return A.EdgeAdding(pe=0.2)
            case "EdgeRemoving":
                return A.EdgeRemoving(pe=0.1)
            case "FeatureMasking":
                return A.FeatureMasking(pf=0.2)
            case "FeatureDropout":
                return A.FeatureDropout(pf=0.2)
            case "EdgeAttrMasking":
                return A.EdgeAttrMasking(pf=0.1)
            case "PPRDiffusion":
                return A.PPRDiffusion()
            case "MDK":
                return A.MarkovDiffusion()
            case "NodeDropping":
                return A.NodeDropping(pn=0.2)
            case "NodeShuffling":
                return A.NodeShuffling()
            case "RWSampling":
                return A.RWSampling()
            case "EgoNet":
                return A.Identity()
            case _:
                raise NameError(f"Unknown augmentation name: {augmentation_name}")

    @staticmethod
    def init_augmentations(augmentation_names, augmentation_strategy):

        if isinstance(augmentation_names, list):
            augmentations = []
            for augmentation_name in augmentation_names:
                augmentations.append(GCLPipeline.init_augmentation(augmentation_name))
            match augmentation_strategy:
                case "Random":
                    return A.RandomChoice(augmentations)
                case "Compose":
                    return A.Compose(augmentations)
        else:
            return GCLPipeline.init_augmentation(augmentation_name)

    @classmethod
    def from_strategy(cls, strategy, device):

        method_name = strategy["method"]

        print(f"##### {method_name} #####")

        architecture_name = strategy["architecture"]
        mode_name = strategy["mode"]
        negative_name = strategy["negative"]
        objective_name = strategy["objective"]

        augmentation1_names = strategy["augmentation1"]
        augmentation1_strategy = strategy["augmentation1_strat"]
        augmentation2_names = strategy["augmentation2"]
        augmentation2_strategy = strategy["augmentation1_strat"]

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
                GCLPipeline.init_augmentations(augmentation1_names, augmentation1_strategy)
                if augmentation1_names is not None
                else None
            ),
            (
                GCLPipeline.init_augmentations(augmentation2_names, augmentation2_strategy)
                if augmentation2_names is not None
                else None
            ),
        ]

        instance = cls(method_name, contrast_model, augmentations, negative_name)

        return instance

    def init_encoder(self, params, device):

        print("Encoder initialization")

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

        print(f"\t input dim: {input_dim}")
        print(f"\t hidden dim: {hidden_dim}")
        print(f"\t # layers: {num_layers}")
        print(f"\t projection dim: {proj_dim}")
        print(f"\t activation: {activation}")

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

    def train(self, dataloader, encoder_model, optimizer):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError()
