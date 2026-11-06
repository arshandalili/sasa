from typing import Callable

import torch
from sae_lens import LoggingConfig
from sae_lens.config import SAETrainerConfig
from sae_lens.training.sae_trainer import SAETrainer


def fit(sae, next_batch: Callable[[], torch.Tensor], steps: int, batch_size: int, lr: float):
    # dead_feature_window = steps: no column counts as dead, so the AuxK loss stays off
    config = SAETrainerConfig(
        n_checkpoints=0, checkpoint_path=None, save_final_checkpoint=False, total_training_samples=steps * batch_size,
        device=sae.W_dec.device.type, autocast=False, lr=lr, lr_end=lr, lr_scheduler_name="constant",
        lr_warm_up_steps=0, adam_beta1=0.9, adam_beta2=0.999, lr_decay_steps=0, n_restart_cycles=1,
        train_batch_size_samples=batch_size, dead_feature_window=steps, feature_sampling_window=steps,
        logger=LoggingConfig(log_to_wandb=False))
    return SAETrainer(config, sae, iter(next_batch, None)).fit()
