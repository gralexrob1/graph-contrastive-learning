import os.path as osp
import json

import GCL.models as M
import GCL.augmentors as A
import GCL.losses as L
from GCL.eval import get_split, LREvaluator

import torch
import torch.nn as nn

from torch.optim import Adam
from torch_geometric.datasets import Planetoid

from gconv import *
from encoders import *


class GCLPipeline:
    def __init__(self, method, contrast_model, augmentations, negative):

        self.method = method
        self.augmentations = augmentations
        self.negative = negative
        self.contrast_model = contrast_model

    @staticmethod
    def init_dataset(dataset_name, data_path, transform=None):
        if dataset_name == "Cora":
            dataset = Planetoid(data_path, dataset_name, transform=transform)
        return dataset

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
                raise NameError(
                    f"Unknown augmentation name: {augmentation_name}")

    @staticmethod
    def init_augmentations(augmentation_names, augmentation_strategy):

        if isinstance(augmentation_names, list):
            augmentations = []
            for augmentation_name in augmentation_names:
                augmentations.append(
                    GCLPipeline.init_augmentation(augmentation_name)
                )
            match augmentation_strategy:
                case "Random": return A.Random(augmentations)
                case "Compose": return A.Compose(augmentations)
        else:
            return GCLPipeline.init_augmentation(augmentation_name)

    @classmethod
    def from_strategy(cls, strategy):

        method_name = strategy["method"]
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
        assert not (
            architecture_name == "WithinEmbedding" and mode_name not in [
                "L2L", "G2G"]
        )

        objective = GCLPipeline.init_objective(objective_name)
        contrast_model = GCLPipeline.init_contrast_model(
            architecture_name,
            objective,
            mode_name,
        )

        augmentations = [
            GCLPipeline.init_augmentations(
                augmentation1_names, augmentation1_strategy
            ) if augmentation1_names is not None else None,
            GCLPipeline.init_augmentations(
                augmentation2_names, augmentation2_strategy
            ) if augmentation2_names is not None else None
        ]

        instance = cls(method_name, contrast_model,
                       augmentations, negative_name)

        return instance

    def init_encoder(self, params):

        input_dim = params["input_dim"]
        hidden_dim = params["hidden_dim"]
        num_layers = params["num_layers"]
        proj_dim = params["proj_dim"]

        # Activation
        if params["activation"] is None:
            activation = None
        else:
            activation = getattr(
                torch.nn, params["activation"],
                ValueError(
                    f"Activation function '{params['activation']}' not found in torch.nn")
            )

        # Device
        device = params["device"]

        len_augmentations = len(self.augmentations)
        augmentor1 = None if not self.augmentations else self.augmentations[0]
        augmentor2 = None if len_augmentations == 1 or augmentor1 is None else self.augmentations[
            1]

        match self.method:

            case "InductiveDGI":
                gconv = DGIInductiveGConv(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
                ).to(device)
                encoder_model = DGIEncoder(
                    encoder=gconv, hidden_dim=hidden_dim
                ).to(device)
                return encoder_model

            case "TransductiveDGI":
                gconv = DGITransductiveGConv(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
                ).to(device)
                encoder_model = DGIEncoder(
                    encoder=gconv, hidden_dim=hidden_dim
                ).to(device)
                return encoder_model

            case "GRACE":
                gconv = GRACEGConv(
                    input_dim=input_dim, hidden_dim=hidden_dim,
                    activation=activation, num_layers=num_layers,
                ).to(device)
                encoder_model = GRACEEncoder(
                    encoder=gconv, augmentor=(augmentor1, augmentor2),
                    hidden_dim=hidden_dim, proj_dim=proj_dim
                ).to(device)
                return encoder_model

            case _:
                raise NotImplementedError

    def train_epoch(self, encoder_model, data, optimizer, device):

        encoder_model.train()
        optimizer.zero_grad()

        match self.method:

            case "TransductiveDGI":
                z, g, zn = encoder_model(data.x, data.edge_index)
                loss = self.contrast_model(h=z, g=g, hn=zn)

            case "GRACE":
                _, z1, z2 = encoder_model(
                    data.x, data.edge_index, data.edge_attr)
                h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
                loss = self.contrast_model(h1, h2)

        loss.backward()
        optimizer.step()

        return loss.item()

    def test(self, encoder_model, data):

        encoder_model.eval()

        z, _, _ = encoder_model(data.x, data.edge_index)
        split = get_split(
            num_samples=z.size()[0],
            train_ratio=0.1, test_ratio=0.8
        )
        result = LREvaluator()(z, data.y, split)

        return result

    def train(self, dataloader, encoder_model, optimizer):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError()
