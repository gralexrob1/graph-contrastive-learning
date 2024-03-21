import os.path as osp
import json

import GCL.models as M
import GCL.augmentors as A
import GCL.losses as L

import torch

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

    def init_dataset(dataset_name, data_path, transform=None):
        if dataset_name == "Cora":
            dataset = Planetoid(data_path, dataset_name, transform=transform)
        return dataset

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

    def init_encoder(self, params):

        input_dim = params["input_dim"]
        hidden_dim = params["hidden_dim"]
        num_layers = params["num_layers"]
        activation = params["activation"]
        proj_dim = params["proj_dim"]
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

    @classmethod
    def from_strategy(cls, strategy):

        method_name = strategy["method"]
        architecture_name = strategy["architecture"]
        mode_name = strategy["mode"]
        augmentation1_name = strategy["augmentation1"]
        augmentation2_name = strategy["augmentation2"]
        negative_name = strategy["negative"]
        objective_name = strategy["objective"]

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

        augmentations = []
        if augmentation1_name is not None:
            augmentation1 = A.RandomChoice(
                [GCLPipeline.init_augmentation(aug)
                 for aug in augmentation1_name],
                1,
            )
            augmentations.append(augmentation1)
        if augmentation2_name is not None:
            augmentation2 = A.RandomChoice(
                [GCLPipeline.init_augmentation(aug)
                 for aug in augmentation2_name],
                1,
            )
            augmentations.append(augmentation2)

        instance = cls(method_name, contrast_model,
                       augmentations, negative_name)

        return instance

    def train(self, dataset, encoder_model, params):

        batch_flag = params["batch_flag"]
        device = params["device"]

    def test(self, encoder_model, dataloader):

        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError()
