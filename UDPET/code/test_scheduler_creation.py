"""Regression check for every scheduler supported by the training CLI."""

from argparse import Namespace

import torch

from model_LeqModGan import modelGAN


opts = Namespace(
    patch_size=[80, 80, 80],
    device=torch.device("cpu"),
    numGPUs=0,
    lr=1e-4,
    preWeight=None,
    weightLoss_localSUVbias=2.0,
    weightLoss_lesionSUVbias=None,
    n_epochs=2,
    gamma=0.1,
    Plateau_step_size=1,
    multi_step_size=[1],
    lr_policy="ReduceLROnPlateau",
)
model = modelGAN(opts)
for policy in ("ReduceLROnPlateau", "multistep", "cosine"):
    opts.lr_policy = policy
    model.set_scheduler(opts)
    if len(model.schedulers) != len(model.optimizers):
        raise AssertionError(f"Scheduler count mismatch for {policy}")
    print("SCHEDULER_OK", policy, [type(item).__name__ for item in model.schedulers])
