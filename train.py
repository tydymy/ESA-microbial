from pathlib import Path

import torch
import sys
sys.path.append("src")


from XXXX_2.config_schema import (ConfigSchema,
                                   DatasetConfigSchemaUniformSampling,
                                   SchedulerConfigSchema, TrainingConfigSchema)
from XXXX_2.dataset import FastaUniformSampler
from XXXX_2.main import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cuda:5")
CONFIG = ConfigSchema(
    training_config=TrainingConfigSchema(
        max_steps=100_000,
        batch_size=16,
        device=device,
        log_interval=100,
        accumulation_steps=16,
        scheduler_config=SchedulerConfigSchema(
            max_lr=1e-4,
        ),
    ),
    dataset_config=DatasetConfigSchemaUniformSampling(
        fasta_file = [Path("s_oneidensis_mr1.fasta")], 
        range_min = 800,
        range_max = 2000,
        subsequence_range_min = 80,
        subsequence_range_max = 180, 
        dataset=FastaUniformSampler,
        sampling_strategy="random_subsequence_uppercase",
        ),
)


main(CONFIG, watch_watch=True)
