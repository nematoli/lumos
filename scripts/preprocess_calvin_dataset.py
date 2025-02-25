import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from lumos.utils.rotation_transformer import RotationTransformer


def resize_image(image, intp, resolution=64):
    """Resize an image to the target size using INTER_AREA interpolation."""
    target_size = (resolution, resolution)
    return cv2.resize(image, target_size, interpolation=intp)


@hydra.main(version_base="1.3", config_path="../config", config_name="preprocess_calvin")
def process_dataset(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    keys = cfg.desired_keys
    img_size = cfg.desired_resolution
    output_dir.mkdir(parents=True, exist_ok=True)

    rot_transform = RotationTransformer(from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")

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

        # Copy lang annotations from original folder to new folder
        orig_lang_annotations = split_path / "lang_annotations" / "auto_lang_ann.npy"

        output_split_lang_path = output_split_path / "lang_annotations"
        output_split_lang_path.mkdir(parents=True, exist_ok=True)
        new_lang_annotations = output_split_lang_path / "auto_lang_ann.npy"
        shutil.copy(orig_lang_annotations, new_lang_annotations)

        # Iterate over .npz files in the directory
        for npz_file in tqdm(split_path.glob("episode_*.npz"), desc=f"Processing {split} data"):
            data = np.load(npz_file)
            extracted_data = {}

            for key in keys:
                # Replace the euler angles of the robot with rotation_6d
                if "robot_obs" in key:
                    robot_obs = data["robot_obs"]
                    euler_angles = robot_obs[3:6]
                    rotation_6d = rot_transform.forward(euler_angles)
                    new_robot_obs = np.concatenate([robot_obs[:3], rotation_6d, robot_obs[6:]])

                # Replace the euler angles of the scene with rotation_6d
                if "scene_obs" in key:
                    scene_obs = data["scene_obs"]
                    # red, blue, pink blocks respectively
                    indices = [[9, 10, 11], [15, 16, 17], [21, 22, 23]]
                    rotation_6ds = []
                    for idx in indices:
                        euler_angles = scene_obs[idx]
                        rotation_6d = rot_transform.forward(euler_angles)
                        rotation_6ds.append(rotation_6d)
                    new_scene_obs = np.concatenate(
                        [
                            scene_obs[:9],
                            rotation_6ds[0],
                            scene_obs[12:15],
                            rotation_6ds[1],
                            scene_obs[18:21],
                            rotation_6ds[2],
                        ],
                    )

                if key in data.keys():
                    if key.startswith("rgb"):
                        extracted_data[key] = resize_image(data[key], intp=cv2.INTER_AREA, resolution=img_size)
                    elif key.startswith("depth"):
                        extracted_data[key] = resize_image(data[key], intp=cv2.INTER_NEAREST, resolution=img_size)
                    elif key == "robot_obs":
                        extracted_data[key] = new_robot_obs
                    elif key == "scene_obs":
                        extracted_data[key] = new_scene_obs
                    else:
                        extracted_data[key] = data[key]
                else:
                    print(f"Key {key} not found in {npz_file}")

            # Prepare the filename for the output file
            output_file = output_split_path / npz_file.name
            np.savez_compressed(output_file, **extracted_data)


if __name__ == "__main__":
    process_dataset()
