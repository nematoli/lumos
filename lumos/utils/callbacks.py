import os
import pytorch_lightning as pl


class WMCustomModelCheckpoint(pl.Callback):
    def __init__(self, savepoints, dirpath, filename="{step}.ckpt"):
        self.savepoints = set(savepoints)
        self.dirpath = dirpath
        self.filename = filename
        self.lowest_val_loss = float("inf")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.savepoints:
            filepath = os.path.join(self.dirpath, self.filename.format(step=step))
            trainer.save_checkpoint(filepath)
            print(f"Checkpoint saved at step {step}: {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        filepath = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(filepath)

        if trainer.sanity_checking:
            return
        current_val_loss = pl_module.running_metrics["loss_total"]

        if current_val_loss is not None and current_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = current_val_loss.clone().detach()
            filepath = os.path.join(self.dirpath, "lowest.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"New minimum val loss detected: {current_val_loss}. Checkpoint saved as: {filepath}")


class ACCustomModelCheckpoint(pl.Callback):
    def __init__(self, savepoints, dirpath, filename="{step}.ckpt"):
        self.savepoints = set(savepoints)
        self.dirpath = dirpath
        self.filename = filename
        self.lowest_val_loss = float("inf")

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     step = trainer.global_step
    #     if step in self.savepoints:
    #         filepath = os.path.join(self.dirpath, self.filename.format(step=step))
    #         trainer.save_checkpoint(filepath)
    #         print(f"Checkpoint saved at step {step}: {filepath}")

    def on_validation_epoch_end(self, trainer, pl_module):
        filepath = os.path.join(self.dirpath, "last.ckpt")
        trainer.save_checkpoint(filepath)
