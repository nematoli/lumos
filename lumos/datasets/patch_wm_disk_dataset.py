from itertools import chain
import logging
import pickle
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from lumos.datasets.base_wm_disk_dataset import BaseWMDiskDataset, load_npz

logger = logging.getLogger(__name__)


class PatchWMDiskDataset(BaseWMDiskDataset):
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
        super().__init__(
            *args,
            reset_prob=reset_prob,
            skip_frames=skip_frames,
            save_format=save_format,
            pretrain=pretrain,
            use_cached_data=False,
            **kwargs,
        )
        # Preloading is different for LIBERO compared to CALVIN
        self.use_cached_data = use_cached_data
        if self.use_cached_data:
            self.preloaded_data = {}  # Initialize as a dictionary
            self.preload_dataset(self.abs_datasets_dir / "cached_data.pkl")

        self.keys = list(chain(*self.observation_space.values()))
        self.key_map = {"patches": "features"}
        self.key_map.update({key: key if key not in self.key_map else self.key_map[key] for key in self.keys})

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

        resets = np.zeros((window_size, 1), dtype=bool)
        if self.reset_prob != 0:
            resets = np.random.rand(window_size, 1) <= self.reset_prob
        zero_action = np.zeros(7, dtype=np.float64)
        zero_action[-1] = 1.0

        if start_idx in self.start_ids:
            episodes = self.zip_sequence(start_idx, end_idx)
            episode = {key: np.stack([ep[self.key_map[key]] for ep in episodes]) for key in self.keys}

            episode["pre_actions"] = np.roll(episode["rel_actions"], shift=1, axis=0)
            episode["pre_actions"][0] = zero_action

            episode["pre_robot_obs"] = np.roll(episode["robot_obs"], shift=1, axis=0)
            resets[0] = True
        else:
            episodes = self.zip_sequence(start_idx - 1, end_idx)
            episode = {key: np.stack([ep[self.key_map[key]] for ep in episodes[1:]]) for key in self.keys}

            episode["pre_actions"] = np.stack([ep["rel_actions"] for ep in episodes[:-1]])

            episode["pre_robot_obs"] = np.stack([ep["robot_obs"] for ep in episodes[:-1]])

        episode["reset"] = resets
        episode["frame"] = np.arange(start_idx, end_idx, dtype=np.int32)[:, np.newaxis]
        return episode

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
                    "rgb_static": np.squeeze(data["rgb_static"]),
                    "rgb_gripper": np.squeeze(data["rgb_gripper"]),
                    "patches": np.squeeze(data["features"]),
                }
                self.preloaded_data[key] = value

            with open(str(cached_data_path), "wb") as f:
                pickle.dump(self.preloaded_data, f)
        logger.info("Preloaded the dataset into cache.")
