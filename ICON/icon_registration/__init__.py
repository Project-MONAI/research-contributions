from icon_registration.losses import (
    GradICON,
    ICON,
    BendingEnergy,
    Diffusion,
    VelocityFieldBendingEnergy,
    VelocityFieldDiffusion,
)

from icon_registration.similarity import (
    LNCC,
    LNCCOnlyInterpolated,
    BlurredSSD,
    SSDOnlyInterpolated,
    SSD,
    NCC,
)
from icon_registration.network_wrappers import (
    InverseConsistentAffine,
    InverseConsistentVelocityField,
    Downsample,
    TwoStep,
    Affine,
    VelocityField,
    DisplacementField,
)
from icon_registration.train import train_batchfunction, train_datasets


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from icon_registration.registration_module import RegistrationModule
