import os
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm


def split_dataset(src_path, dest_path, val_ratio=0.1):
    """
    Split the dataset into training and validation sets.
    """
    # Get episode start and end ids
    ep_start_end_ids = np.load(src_path / "ep_start_end_ids.npy")
    ep_lens = np.load(src_path / "ep_lens.npy")
    ann = np.load(src_path / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).reshape(-1)[0]
    np.random.seed(42)
    train_indices = np.random.choice(
        len(ep_start_end_ids), size=int(len(ep_start_end_ids) * (1 - val_ratio)), replace=False
    )

    for split in tqdm(["training", "validation"]):
        split_path = dest_path / split
        split_path.mkdir(parents=True, exist_ok=True)

        ep_start_end_ids_split = []
        ep_lens_split = []
        ann_split = {
            "language": {"ann": [], "task": [], "emb": []},
            "info": {"episodes": [], "indx": []},
        }

        if split == "training":
            indices = sorted(train_indices)
        else:
            indices = np.setdiff1d(np.arange(len(ep_start_end_ids)), train_indices)

        for i in indices:
            ep_start_end_ids_split.append(ep_start_end_ids[i])
            ep_lens_split.append(ep_lens[i])
            ann_split["language"]["task"].append(ann["language"]["task"][i])
            ann_split["language"]["ann"].append(ann["language"]["ann"][i])
            ann_split["info"]["indx"].append(ann["info"]["indx"][i])

        np.save(split_path / "ep_start_end_ids.npy", np.array(ep_start_end_ids_split))
        np.save(split_path / "ep_lens.npy", np.array(ep_lens_split))
        np.save(split_path / "lang_annotations" / "auto_lang_ann.npy", ann_split)

        # Copy the source dataset to the destination as is
        for start, end in tqdm(ep_start_end_ids_split):
            for i in range(start, end + 1):
                src_file = src_path / f"episode_{i:06d}.npz"
                dest_file = split_path / f"episode_{i:06d}.npz"
                shutil.copy(src_file, dest_file)


if __name__ == "__main__":
    src_path = Path("/path/to/your/source/dataset")
    dest_path = Path("/path/to/your/destination/dataset")

    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(dest_path / "training", exist_ok=True)
    os.makedirs(dest_path / "validation", exist_ok=True)
    os.makedirs(dest_path / "training" / "lang_annotations", exist_ok=True)
    os.makedirs(dest_path / "validation" / "lang_annotations", exist_ok=True)
    split_dataset(src_path, dest_path, val_ratio=0.05)
