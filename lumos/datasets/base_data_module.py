import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision

from lumos.datasets.utils.episode_utils import load_dataset_statistics
from lumos.utils.nn_utils import (
    transpose_collate_ag,
    transpose_collate_hybrid_wm,
    transpose_collate_state_wm,
    transpose_collate_wm,
)

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        training_repo_root: Optional[Path] = None,
        root_data_dir: str = "datasets/task_D_D",
        transforms: DictConfig = DEFAULT_TRANSFORM,
        batch_sampler: DictConfig = None,
        shuffle_val: bool = False,
        load_feats: bool = False,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            assert training_repo_root is not None, "If root_data_path isn't absolute, please provide training_repo_root"
            root_data_path = training_repo_root / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.batch_sampler = batch_sampler
        self.load_feats = load_feats
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        if self.datasets_cfg.wm_disk_dataset.key == "vis":
            self.collate_fn = transpose_collate_wm
        elif self.datasets_cfg.wm_disk_dataset.key == "hybrid":
            self.collate_fn = transpose_collate_hybrid_wm
        elif self.datasets_cfg.wm_disk_dataset.key == "state":
            self.collate_fn = transpose_collate_state_wm

    def _compute_batch_params(self):
        """Return (num_replicas, global_batch_size)."""
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size, self.train_batch_size * world_size, self.val_batch_size * world_size

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            logger.error(
                f"""No dataset found in {Path(self.training_dir).parent}.
                Please make sure you set the correct dataset path.
                For information how to download one of the CALVIN datasets, please visit
                https://github.com/mees/calvin/tree/main/dataset"""
            )
            exit()

    def setup(self, stage=None):
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        self.train_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        }

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.val_datasets = {}, {}

        if self.load_feats:
            train_features = self.load_latent_dataset(self.training_dir / "cached_feats.pkl")
            val_features = self.load_latent_dataset(self.val_dir / "cached_feats.pkl")

        for _, dataset in self.datasets_cfg.items():
            train_dataset = hydra.utils.instantiate(
                dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
            )
            val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)

            if self.load_feats:
                train_dataset.setup_features(train_features)
                val_dataset.setup_features(val_features)

            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.val_datasets[key] = val_dataset
            self.modalities.append(key)

    def train_dataloader(self):
        num_replicas, global_train_batch_size, global_val_batch_size = self._compute_batch_params()

        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = self.collate_fn
        if self.batch_sampler == {}:
            return {
                key: DataLoader(
                    dataset,
                    batch_size=self.train_batch_size,
                    num_workers=dataset.num_workers,
                    pin_memory=False,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                for key, dataset in self.train_datasets.items()
            }
        else:
            dataloaders = {}
            for key, dataset in self.train_datasets.items():
                dataset_length = len(dataset)
                print(f"Dataset '{key}' has {dataset_length} items.")  # Or use another form of logging if preferred

                if num_replicas > 1:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.global_batch_size = global_train_batch_size
                    self.batch_sampler.num_replicas = num_replicas
                else:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.batch_size = self.train_batch_size
                batch_sampler = hydra.utils.instantiate(self.batch_sampler)

                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=dataset.num_workers,
                    pin_memory=True,
                    collate_fn=collate,
                    persistent_workers=True,
                )
                dataloaders[key] = dataloader
            return dataloaders

    def val_dataloader(self):
        num_replicas, global_train_batch_size, global_val_batch_size = self._compute_batch_params()

        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = self.collate_fn
        if self.batch_sampler == {}:
            return {
                key: DataLoader(
                    dataset,
                    batch_size=self.val_batch_size,
                    num_workers=dataset.num_workers,
                    pin_memory=True,
                    shuffle=self.shuffle_val,
                    collate_fn=collate,
                    persistent_workers=True,
                )
                for key, dataset in self.val_datasets.items()
            }
        else:
            val_dataloaders = {}
            for key, dataset in self.val_datasets.items():
                dataset_length = len(dataset)
                print(f"Dataset '{key}' has {dataset_length} items.")  # Or use another form of logging if preferred

                if num_replicas > 1:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.global_batch_size = global_val_batch_size
                    self.batch_sampler.num_replicas = num_replicas
                else:
                    self.batch_sampler.data_size = dataset_length
                    self.batch_sampler.batch_size = self.val_batch_size
                batch_sampler = hydra.utils.instantiate(self.batch_sampler)

                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=dataset.num_workers,
                    pin_memory=False,
                    shuffle=self.shuffle_val,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                val_dataloaders[key] = dataloader

            return val_dataloaders

    def load_latent_dataset(self, features_path):
        if features_path.is_file():
            logger.info("Loading preloaded features from cache...")
            with open(str(features_path), "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Cached features not found. Generate features via featurizer.py")
