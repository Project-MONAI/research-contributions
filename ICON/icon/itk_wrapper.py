import copy

import itk
import numpy as np
import torch
import torch.nn.functional as F

from icon_registration import config
from icon_registration.losses import to_floats


def finetune_execute(model, image_A, image_B, steps):
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.load_state_dict(state_dict)
    return loss


def register_pair(
    model, image_A, image_B, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_resized, B_resized)
    else:
        loss = finetune_execute(model, A_resized, B_resized, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)

def register_pair_with_multimodalities(
    model, image_A: list, image_B: list, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    assert len(image_A) == len(image_B), "image_A and image_B should have the same number of modalities."

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device)

    A_npy, B_npy = [], []
    for image_a, image_b in zip(image_A, image_B):
        assert isinstance(image_a, itk.Image)
        assert isinstance(image_b, itk.Image)

        A_npy.append(np.array(image_a))
        B_npy.append(np.array(image_b))

        assert(np.max(A_npy[-1]) != np.min(A_npy[-1]))
        assert(np.max(B_npy[-1]) != np.min(B_npy[-1]))

    # turn images into torch Tensors: add batch dimensions (each of length 1)
    A_trch = torch.Tensor(np.array(A_npy)).to(config.device)[None]
    B_trch = torch.Tensor(np.array(B_npy)).to(config.device)[None]

    shape = model.identity_map.shape[2:]
    if list(A_trch.shape[2:]) != list(shape) or (list(B_trch.shape[2:]) != list(shape)):
        # Here we resize the input images to the shape expected by the neural network. This affects the
        # pixel stride as well as the magnitude of the displacement vectors of the resulting
        # displacement field, which create_itk_transform will have to compensate for.
        A_trch = F.interpolate(
            A_trch, size=shape, mode="trilinear", align_corners=False
        )
        B_trch = F.interpolate(
            B_trch, size=shape, mode="trilinear", align_corners=False
        )

    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_trch, B_trch)
    else:
        loss = finetune_execute(model, A_trch, B_trch, finetune_steps)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_trch, B_trch)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        create_itk_transform(phi_AB, model.identity_map, image_A[0], image_B[0]),
        create_itk_transform(phi_BA, model.identity_map, image_B[0], image_A[0]),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)


def create_itk_transform(phi, ident, image_A, image_B) -> "itk.CompositeTransform":

    # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
    disp = (phi - ident)[0].cpu()

    network_shape_list = list(ident.shape[2:])

    dimension = len(network_shape_list)

    tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()

    # We convert the displacement field into an itk Vector Image.
    scale = torch.Tensor(network_shape_list)

    for _ in network_shape_list:
        scale = scale[:, None]
    disp *= scale - 1

    # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
    disp_itk_format = (
        disp.double()
        .numpy()[list(reversed(range(dimension)))]
        .transpose(list(range(1, dimension + 1)) + [0])
    )
    # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
    # as expected by itk.

    itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)

    tr.SetDisplacementField(itk_disp_field)

    to_network_space = resampling_transform(image_A, list(reversed(network_shape_list)))

    from_network_space = resampling_transform(
        image_B, list(reversed(network_shape_list))
    ).GetInverseTransform()

    phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()

    phi_AB_itk.PrependTransform(from_network_space)
    phi_AB_itk.PrependTransform(tr)
    phi_AB_itk.PrependTransform(to_network_space)

    # warp(image_A, phi_AB_itk) is close to image_B

    return phi_AB_itk


def resampling_transform(image, shape):

    imageType = itk.template(image)[0][itk.template(image)[1]]

    dummy_image = itk.image_from_array(
        np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype)
    )
    if len(shape) == 2:
        transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]
    else:
        transformType = itk.VersorRigid3DTransform[itk.D]
    initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    initializer = initType.New()
    initializer.SetFixedImage(dummy_image)
    initializer.SetMovingImage(image)
    transform = transformType.New()

    initializer.SetTransform(transform)
    initializer.InitializeTransform()

    if len(shape) == 3:
        transformType = itk.CenteredAffineTransform[itk.D, 3]
        t2 = transformType.New()
        t2.SetCenter(transform.GetCenter())
        t2.SetOffset(transform.GetOffset())
        transform = t2
    m = transform.GetMatrix()
    m_a = itk.array_from_matrix(m)

    input_shape = image.GetLargestPossibleRegion().GetSize()

    for i in range(len(shape)):

        m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])

    m_a = itk.array_from_matrix(image.GetDirection()) @ m_a

    transform.SetMatrix(itk.matrix_from_array(m_a))

    return transform
