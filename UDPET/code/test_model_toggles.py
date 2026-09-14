from argparse import Namespace

import torch

from model_LeqModGan import modelGAN


def options(qumod, lemod):
    return Namespace(
        patch_size=[80, 80, 80],
        device=torch.device("cpu"),
        numGPUs=0,
        lr=1e-4,
        preWeight=None,
        weightLoss_localSUVbias=2.0 if qumod else None,
        weightLoss_lesionSUVbias=2.0 if lemod else None,
    )


for label, qumod, lemod in (
    ("BOTH_OFF", False, False),
    ("QUMOD_ON_LEMOD_OFF", True, False),
    ("QUMOD_OFF_LEMOD_ON", False, True),
):
    model = modelGAN(options(qumod, lemod))
    print(label, model.loss_names)
    del model
