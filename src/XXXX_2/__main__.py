"""
cli interface for training the contrastive self-supervised sequence model
"""


from pathlib import Path
from typing import Optional

import typer
import wandb

from XXXX-2.model import Encoder
from XXXX-2.trainer import ContrastiveTrainer
from XXXX-2.utils import cfg_to_wandb_dict, get_config_from_path


def main(config_path: Optional[str] = None):
    """
    Args:
        cfg: Path to the configuration file, should be a .py file containing a Config instance
    """
    if config_path is None:
        cfg = Path(__file__).parents[2] / "configs" / "default_config.py"
    else:
        cfg = Path(config_path)

    config = get_config_from_path(cfg)
    # log config to wandb
    wandb.init(project="XXXX-2", config=cfg_to_wandb_dict(config))

    model_cfg = config.model_config
    training_cfg = config.training_config

    model = Encoder(**model_cfg.dict())

    optimizer_cfg = training_cfg.optimizer_config
    optimizer = training_cfg.optimizer(model.parameters(), **optimizer_cfg.dict())

    trainer = ContrastiveTrainer(
        model=model,
        criterion=training_cfg.criterion,
        optimizer=optimizer,
        train_dataloader=training_cfg.train_dataloader,
        device=training_cfg.device,
    )

    trainer.train(
        max_steps=training_cfg.max_steps,
        log_interval=training_cfg.log_interval,
    )


if __name__ == "__main__":
    typer.run(main)
