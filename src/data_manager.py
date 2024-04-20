import glob
import math
import os

import numpy as np

import torch
from torch.utils.data import Dataset


class LRPPILabeledDataset(Dataset):
    """
    No further transform since dataset is already processed.
    """

    def __init__(self, npy_file, processed_dir):
        self.npy_ar = np.load(npy_file)
        self.processed_dir = processed_dir
        self.protein_1 = self.npy_ar[:, 2]
        self.protein_2 = self.npy_ar[:, 5]
        self.label = self.npy_ar[:, 6].astype(float)
        self.n_samples = self.npy_ar.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        prot_1_file = os.path.join(self.processed_dir, self.protein_1[index] + ".pt")
        prot_2_file = os.path.join(self.processed_dir, self.protein_2[index] + ".pt")
        prot_1 = torch.load(glob.glob(prot_1_file)[0])
        prot_2 = torch.load(glob.glob(prot_2_file)[0])

        return prot_1, prot_2, torch.tensor(self.label[index])
