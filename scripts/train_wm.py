import logging
from pathlib import Path
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import rank_zero_only
from lumos.utils.info_utils import print_system_env_info, get_last_checkpoint, setup_logger, setup_callbacks

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="train_wm")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """
    if cfg.exp_dir is None:
        cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)
    log_rank_0(f"Training a World Model with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    chk = get_last_checkpoint(model_dir)

    if chk is not None:
        if cfg.world_model.name == "dreamer_v2":
            from lumos.world_models.dreamer_v2 import DreamerV2

            model = DreamerV2.load_from_checkpoint(chk.as_posix())
        else:
            raise NotImplementedError(f"Unknown model: {cfg.world_model.name}")
    else:
        model = hydra.utils.instantiate(cfg.world_model)

    trainer_args = {**cfg.trainer, "logger": setup_logger(cfg), "callbacks": setup_callbacks(cfg.callbacks)}
    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=datamodule, ckpt_path=chk)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """
    Log the information using the logger at rank 0.
    """
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
