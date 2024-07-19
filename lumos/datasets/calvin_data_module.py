import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from lumos.datasets.utils.episode_utils import load_dataset_statistics
from lumos.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
import torchvision
from lumos.utils.nn_utils import transpose_collate_wm, transpose_collate_ag
import pickle

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})


class CalvinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        training_repo_root: Optional[Path] = None,
        root_data_dir: str = "datasets/task_D_D",
        transforms: DictConfig = DEFAULT_TRANSFORM,
        batch_sampler: DictConfig = None,
        shuffle_val: bool = False,
        load_feats: bool = False,
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

        self.use_shm = "shm_dataset" in self.datasets_cfg.vision_dataset._target_

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

        if self.use_shm:
            # When using shared memory dataset, initialize lookups
            train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

            val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_shm_lookup(train_shm_lookup, val_shm_lookup)

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

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()

        if self.load_feats:
            train_features = self.load_latent_dataset(self.training_dir / "cached_feats.pkl")
            val_features = self.load_latent_dataset(self.val_dir / "cached_feats.pkl")

        for _, dataset in self.datasets_cfg.items():
            train_dataset = hydra.utils.instantiate(
                dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
            )
            val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)
            if self.use_shm:
                train_dataset.setup_shm_lookup(train_shm_lookup)
                val_dataset.setup_shm_lookup(val_shm_lookup)

            if self.load_feats:
                train_dataset.setup_features(train_features)
                val_dataset.setup_features(val_features)

            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.val_datasets[key] = val_dataset
            self.modalities.append(key)

    def train_dataloader(self):
        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = transpose_collate_wm
        if self.batch_sampler == {}:
            return {
                key: DataLoader(
                    dataset,
                    batch_size=dataset.batch_size,
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

                self.batch_sampler.data_size = dataset_length
                batch_sampler = hydra.utils.instantiate(self.batch_sampler)

                dataloader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=dataset.num_workers,
                    pin_memory=False,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                dataloaders[key] = dataloader
            return dataloaders

    def val_dataloader(self):
        if self.load_feats:
            collate = transpose_collate_ag
        else:
            collate = transpose_collate_wm
        if self.batch_sampler == {}:
            val_dataloaders = {
                key: DataLoader(
                    dataset,
                    batch_size=dataset.batch_size,
                    num_workers=dataset.num_workers,
                    pin_memory=False,
                    shuffle=self.shuffle_val,
                    collate_fn=collate,
                    persistent_workers=False,
                )
                for key, dataset in self.val_datasets.items()
            }
            combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
            return combined_val_loaders
        else:
            val_dataloaders = {}
            for key, dataset in self.val_datasets.items():
                dataset_length = len(dataset)
                print(f"Dataset '{key}' has {dataset_length} items.")
                self.batch_sampler.data_size = dataset_length
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

            combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
            return combined_val_loaders

    def load_latent_dataset(self, features_path):
        if features_path.is_file():
            logger.info("Loading preloaded features from cache...")
            with open(str(features_path), "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Cached features not found. Generate features via featurizer.py")
