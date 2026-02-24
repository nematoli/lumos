"""
This script is used to prepare action chunks for the Calvin dataset.
For each frame, the script creates action chunks of size action_chunk_size. 
The chunks are "pre-actions" i.e. the actions that lead to the current frame. 

The file saved is a npz file with the only key "rel_actions".

The dataset is saved in CALVIN format, i.e. in the same folder structure as the original dataset.
"""
from pathlib import Path
import shutil

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm



@hydra.main(version_base="1.3", config_path="../config", config_name="create_action_chunks_calvin")
def process_dataset(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    action_chunk_size = cfg.action_chunk_size
    n_digits = cfg.n_digits

    output_dir.mkdir(parents=True, exist_ok=True)
    """Process the Calvin dataset and create a smaller version of it."""
    for split in ["training", "validation"]:
        split_path = Path(input_dir) / split
        output_split_path = Path(output_dir) / split
        output_split_path.mkdir(parents=True, exist_ok=True)

        # Load episode start and end ids if needed for processing
        ep_start_end_ids = np.load(split_path / "ep_start_end_ids.npy", allow_pickle=True)

        # Copy ep_start_end_ids.npy from original folder to new folder
        orig_ep_start_end_ids = split_path / "ep_start_end_ids.npy"
        new_ep_start_end_ids = output_split_path / "ep_start_end_ids.npy"

        shutil.copy(orig_ep_start_end_ids, new_ep_start_end_ids)

        # Iterate over each frame in the episode
        for ep_start, ep_end in tqdm(ep_start_end_ids, desc="Processing episodes"):
            action_chunk = np.zeros((action_chunk_size, 7))
            output_file = output_split_path / f"episode_{ep_start:0{n_digits}d}.npz"
            np.savez_compressed(output_file, rel_actions=action_chunk.reshape(-1))
            for ep_idx in tqdm(range(ep_start+1, ep_end), desc="Processing pre-action chunks"):
                action_chunk = np.roll(action_chunk, shift=-1, axis=0)
                action_chunk[-1] = np.load(split_path / f"episode_{ep_idx-1:0{n_digits}d}.npz")["rel_actions"]
                output_file = output_split_path / f"episode_{ep_idx:0{n_digits}d}.npz"
                np.savez_compressed(output_file, rel_actions=action_chunk.reshape(-1))

if __name__ == "__main__":
    process_dataset()
