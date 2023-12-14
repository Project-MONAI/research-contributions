from os.path import exists

import torch

import icon_registration
from icon_registration import config

from .. import networks
from .lung_ct import init_network
from ..losses import SSD


def OAI_knees_registration_model(pretrained=True):
    # The definition of our final 4 step registration network.

    phi = icon_registration.FunctionFromVectorField(
        networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
    )
    psi = icon_registration.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    pretrained_lowres_net = icon_registration.TwoStepRegistration(phi, psi)

    hires_net = icon_registration.TwoStepRegistration(
        icon_registration.DownsampleRegistration(pretrained_lowres_net, dimension=3),
        icon_registration.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
    )

    fourth_net = icon_registration.InverseConsistentNet(
        icon_registration.TwoStepRegistration(
            hires_net,
            icon_registration.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        ),
        SSD(),
        3600,
    )

    BATCH_SIZE = 3
    SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
    input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

    if pretrained:
        weights_location = "network_weights/pretrained_OAI_model_compact"
        if not exists(weights_location):
            print("Downloading pretrained model (500 Mb)")
            import urllib.request
            import os

            os.makedirs("network_weights", exist_ok=True)

            urllib.request.urlretrieve(
                "https://github.com/uncbiag/ICON/releases/download/pretrained_oai_model/OAI_knees_ICON_model.pth",
                weights_location,
            )

        trained_weights = torch.load(weights_location, map_location=torch.device("cpu"))
        fourth_net.load_state_dict(trained_weights)

    fourth_net.assign_identity_map(input_shape)

    net = fourth_net
    net.to(config.device)
    net.eval()
    return net


def OAI_knees_gradICON_model(pretrained=True):
    return init_network("knee", pretrained)
