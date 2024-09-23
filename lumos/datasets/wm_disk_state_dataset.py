from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

from lumos.datasets.base_dataset import BaseDataset
from lumos.datasets.utils.episode_utils import lookup_naming_pattern
import numpy as np
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class WMDiskStateDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        reset_prob: float = 0.05,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        use_cached_data: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.reset_prob = reset_prob
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        self.episode_lookup, self.start_ids = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)
        self.use_cached_data = use_cached_data
        if self.use_cached_data:
            self.preload_dataset(self.abs_datasets_dir / "cached_data.pkl")

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        # keys.append("scene_obs")

        resets = np.zeros((window_size, 1), dtype=bool)
        if self.reset_prob != 0:
            resets = np.random.rand(window_size, 1) <= self.reset_prob
        zero_action = np.zeros(7, dtype=np.float64)
        zero_action[-1] = 1.0

        if start_idx in self.start_ids:
            episodes = self.zip_sequence(start_idx, end_idx)
            # [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
            episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}

            episode["pre_actions"] = np.roll(episode["rel_actions"], shift=1, axis=0)
            episode["pre_actions"][0] = zero_action

            episode["pre_robot_obs"] = np.roll(episode["robot_obs"], shift=1, axis=0)
            resets[0] = True
        else:
            episodes = self.zip_sequence(start_idx - 1, end_idx)

            # [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx - 1, end_idx)]
            episode = {key: np.stack([ep[key] for ep in episodes[1:]]) for key in keys}

            episode["pre_actions"] = np.stack([ep["rel_actions"] for ep in episodes[:-1]])

            episode["pre_robot_obs"] = np.stack([ep["robot_obs"] for ep in episodes[:-1]])

        # reset_indices = np.nonzero(resets.squeeze())[0]
        # for idx in reset_indices:
        #     episode["pre_actions"][idx] = zero_action

        episode["reset"] = resets
        episode["frame"] = np.arange(start_idx, end_idx, dtype=np.int32)[:, np.newaxis]
        return episode

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        start_ids = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            start_ids.append(start_idx)
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup), start_ids

    def preload_dataset(self, cached_data_path):
        """Preloads the entire dataset into memory."""
        if cached_data_path.is_file():
            logger.info("Loading preloaded data from cache...")
            with open(str(cached_data_path), "rb") as f:
                self.preloaded_data = pickle.load(f)
        else:
            data_dir_list = sorted([item for item in self.abs_datasets_dir.iterdir()])
            for file_path in tqdm(data_dir_list, desc="Preloading dataset"):
                if "npz" not in file_path.suffix:
                    continue

                key = self.extract_episode_number(file_path)
                np_obj = load_npz(file_path)
                data = {key: np.stack([np_obj[key]]) for key, _ in np_obj.items()}

                value = {
                    "rel_actions": np.squeeze(data["rel_actions"]),
                    "robot_obs": np.squeeze(data["robot_obs"]),
                    "scene_obs": np.squeeze(data["scene_obs"]),
                }
                self.preloaded_data[key] = value

            with open(str(cached_data_path), "wb") as f:
                pickle.dump(self.preloaded_data, f)
        logger.info("Preloaded the dataset into cache.")

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        if not self.use_cached_data:
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        else:
            episodes = [self.preloaded_data[file_idx] for file_idx in range(start_idx, end_idx)]

        return episodes

    def extract_episode_number(self, file_path):
        # Regular expression to find the episode number pattern
        match = re.search(r"episode_(\d+)\.npz", str(file_path))
        if match:
            # Return the episode number as an integer
            return int(match.group(1))
        else:
            raise ValueError(f"Episode number not found in {file_path}")
