from pathlib import Path
import shutil

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../config", config_name="append_calvin_patch")
def process_dataset(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    keys = cfg.desired_keys
    """Process the Calvin dataset and create a smaller version of it."""
    for split in ["training", "validation"]:
        output_split_path = Path(output_dir) / split
        input_split_path = Path(input_dir) / split

        # Copy ep_start_end_ids.npy from original folder to new folder
        orig_ep_start_end_ids = input_split_path / "ep_start_end_ids.npy"
        new_ep_start_end_ids = output_split_path / "ep_start_end_ids.npy"

        shutil.copy(orig_ep_start_end_ids, new_ep_start_end_ids)

        # Iterate over .npz files in the directory
        for out_npz_file in tqdm(output_split_path.glob("episode_*.npz"), desc=f"Processing {split} data"):
            out_data = dict(np.load(out_npz_file))
            input_npz_file = input_split_path / out_npz_file.name
            input_data = np.load(input_npz_file)

            for key in keys:
                if key in input_data:
                    out_data[key] = input_data[key]
                else:
                    print(f"Key {key} not found in {input_npz_file.name}")

            # Prepare the filename for the output file
            output_file = output_split_path / out_npz_file.name
            np.savez_compressed(output_file, **out_data)


if __name__ == "__main__":
    process_dataset()
