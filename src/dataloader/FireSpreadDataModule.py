from pathlib import Path

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional

class FireSpreadDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool, 
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False, 
                 add_binary_fire_mask: bool = True, *args, **kwargs):
        super().__init__()

        self.add_binary_fire_mask = add_binary_fire_mask
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: str):
        train_years, val_years, test_years = self.split_fires()
        self.train_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=train_years,
                                               n_leading_observations=self.n_leading_observations,
                                               crop_side_length=self.crop_side_length,
                                               load_from_hdf5=self.load_from_hdf5, is_train=True,
                                               remove_duplicate_features=self.remove_duplicate_features,
                                               features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                               add_binary_fire_mask=self.add_binary_fire_mask)
        self.val_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=val_years,
                                             n_leading_observations=self.n_leading_observations,
                                             crop_side_length=self.crop_side_length,
                                             load_from_hdf5=self.load_from_hdf5, is_train=True,
                                             remove_duplicate_features=self.remove_duplicate_features,
                                             features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                             add_binary_fire_mask=self.add_binary_fire_mask)
        self.test_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=test_years,
                                              n_leading_observations=self.n_leading_observations,
                                              crop_side_length=self.crop_side_length,
                                              load_from_hdf5=self.load_from_hdf5, is_train=False,
                                              remove_duplicate_features=self.remove_duplicate_features, 
                                              features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                              add_binary_fire_mask=self.add_binary_fire_mask)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def split_fires(self):
        train_years = [2018, 2020]
        val_years = [2019]
        test_years = [2021]

        return train_years, val_years, test_years

