import itk
import torch

import icon_registration.config as config

from .. import losses, network_wrappers, networks


def make_network():
    dimension = 3
    inner_net = network_wrappers.FunctionFromVectorField(
        networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = network_wrappers.TwoStepRegistration(
            network_wrappers.DownsampleRegistration(inner_net,
                                                    dimension=dimension),
            network_wrappers.FunctionFromVectorField(
                networks.tallUNet2(dimension=dimension)))
    inner_net = network_wrappers.TwoStepRegistration(
        inner_net,
        network_wrappers.FunctionFromVectorField(
            networks.tallUNet2(dimension=dimension)))

    net = losses.GradientICONSparse(inner_net,
                              similarity=losses.LNCC(sigma=5),
                              lmbda=1.5)

    return net


def init_network(task, pretrained=True):
    if task == "lung":
        input_shape = [1, 1, 175, 175, 175]
    elif task == "knee":
        input_shape = [1, 1, 80, 192, 192]
    elif task == "brain":
        input_shape = [1, 1, 130, 155, 130]
    else:
        print(f"Task {task} is not defined. Fall back to the lung model.")
        task = "lung"
        input_shape = [1, 1, 175, 175, 175]

    net = make_network()
    net.assign_identity_map(input_shape)

    if pretrained:
        from os.path import exists
        weights_location = f"network_weights/{task}_model"

        if not exists(f"{weights_location}/{task}_model_weights.trch"):
            print("Downloading pretrained model")
            import urllib.request
            import os
            download_path = "https://github.com/uncbiag/ICON/releases/download"
            download_path = f"{download_path}/pretrained_models_v1.0.0"

            os.makedirs(weights_location, exist_ok=True)
            urllib.request.urlretrieve(
                f"{download_path}/{task}_model_weights_step_2.trch",
                f"{weights_location}/{task}_model_weights.trch",
            )

        trained_weights = torch.load(
            f"{weights_location}/{task}_model_weights.trch",
            map_location=torch.device("cpu"),
        )
        net.regis_net.load_state_dict(trained_weights, strict=False)
    net.assign_identity_map(input_shape)

    net.to(config.device)
    net.eval()
    return net


def lung_network_preprocess(image: "itk.Image",
                            segmentation: "itk.Image") -> "itk.Image":

    image = itk.clamp_image_filter(image, Bounds=(-1000, 0))
    cast_filter = itk.CastImageFilter[type(image), itk.Image.F3].New()
    cast_filter.SetInput(image)
    cast_filter.Update()
    image = cast_filter.GetOutput()

    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                                   itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()

    image = itk.shift_scale_image_filter(image, shift=1000, scale=1 / 1000)

    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                          itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()


def LungCT_registration_model(pretrained=True):
    return init_network("lung", pretrained=pretrained)
