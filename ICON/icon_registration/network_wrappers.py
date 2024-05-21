import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from icon_registration.registration_module import RegistrationModule


class DisplacementField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        concatenated_images = torch.cat([image_A, image_B], axis=1)
        tensor_of_displacements = self.net(concatenated_images)
        displacement_field = self.as_function(tensor_of_displacements)

        def transform(coordinates):
            return coordinates + displacement_field(coordinates)

        return {"phi_AB": transform}


class VelocityField(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 256

    def forward(self, image_A, image_B):
        concatenated_images = torch.cat([image_A, image_B], axis=1)
        velocity_field = self.net(concatenated_images)
        velocityfield_delta = velocity_field / self.n_steps

        for _ in range(8):
            velocityfield_delta = velocityfield_delta + self.as_function(
                velocityfield_delta
            )(velocityfield_delta + self.identity_map)

        def transform(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta
            )(coordinate_tensor)
            return coordinate_tensor

        return {"phi_AB": transform, "velocity_fields": [velocity_field]}


def multiply_matrix_vectorfield(matrix, vectorfield):
    dimension = len(vectorfield.shape) - 2
    if dimension == 2:
        batch_matrix_multiply = "ijkl,imj->imkl"
    else:
        batch_matrix_multiply = "ijkln,imj->imkln"
    return torch.einsum(batch_matrix_multiply, vectorfield, matrix)


class Affine(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B)

        def transform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return multiply_matrix_vectorfield(matrix_phi, coordinates_homogeneous)[
                :, :-1
            ]

        return {"phi_AB": transform}


class TwoStep(RegistrationModule):
    """Combine two RegistrationModules.

    First netPhi is called on the input images, then image_A is warped with
    the resulting field, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        phi = self.netPhi(image_A, image_B)
        psi = self.netPsi(self.as_function(image_A)(phi["phi_AB"](self.identity_map)), image_B)
        result = {
            "phi_AB": lambda tensor_of_coordinates: phi["phi_AB"](
                psi["phi_AB"](tensor_of_coordinates)
            )
        }

        regularization_loss = 0
        if "regularization_loss" in phi:
            regularization_loss += phi["regularization_loss"]
        if "regularization_loss" in psi:
            regularization_loss += psi["regularization_loss"]
        if "regularization_loss" in phi or "regularization_loss" in psi:
            result["regularization_loss"] = regularization_loss

        velocity_fields = []
        if "velocity_fields" in phi:
            velocity_fields += phi["regularization_loss"]
        if "velocity_fields" in psi:
            velocity_fields += psi["regularization_loss"]
        if "velocity_fields" in phi or "regularization_loss" in psi:
            result["velocity_fields"] = regularization_loss
        return result


class Downsample(RegistrationModule):
    """
    Perform registration using the wrapped RegistrationModule `net`
    at half input resolution.
    """

    def __init__(self, net, dimension):
        super().__init__()
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        # This member variable is read by assign_identity_map when
        # walking the network tree and assigning identity_maps
        # to know that all children of this module operate at a lower
        # resolution.
        self.downscale_factor = 2

    def forward(self, image_A, image_B):
        image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        result = self.net(image_A, image_B)

        # MONAI's ddf coordinate convention depends on resolution:

        for key in ["phi_AB", "phi_BA"]:
            if key in lowres_result:
                highres_phi = lambda coords: 2 * result[key](coords / 2)
                result[key] = highres_phi

        return result


class InverseConsistentVelocityField(RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.n_steps = 7

    def forward(self, image_A, image_B):
        concatenated_images_AB = torch.cat([image_A, image_B], axis=1)
        concatenated_images_BA = torch.cat([image_B, image_A], axis=1)
        velocity_field = self.net(concatenated_images_AB) - self.net(
            concatenated_images_BA
        )
        velocityfield_delta_ab = velocity_field / 2**self.n_steps
        velocityfield_delta_ba = -velocityfield_delta_ab

        for _ in range(self.n_steps):
            velocityfield_delta_ab = velocityfield_delta_ab + self.as_function(
                velocityfield_delta_ab
            )(velocityfield_delta_ab + self.identity_map)

        def transform_AB(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta_ab
            )(coordinate_tensor)
            return coordinate_tensor

        for _ in range(self.n_steps):
            velocityfield_delta_ba = velocityfield_delta_ba + self.as_function(
                velocityfield_delta_ba
            )(velocityfield_delta_ba + self.identity_map)

        def transform_BA(coordinate_tensor):
            coordinate_tensor = coordinate_tensor + self.as_function(
                velocityfield_delta_ba
            )(coordinate_tensor)
            return coordinate_tensor

        return {
            "phi_AB": transform_AB,
            "phi_BA": transform_BA,
            "velocity_fields": [velocity_field],
        }


class InverseConsistentAffine(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an Batch x N*N+1 tensor representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        concatenated_images_AB = torch.cat([image_A, image_B], axis=1)
        concatenated_images_BA = torch.cat([image_B, image_A], axis=1)
        matrix_phi = self.net(concatenated_images_AB) - self.net(concatenated_images_BA)

        matrix_phi = matrix_phi.reshape(
            image_A.shape[0], len(image_A.shape), len(image_A.shape) + 1
        )

        matrix_phi_AB = torch.linalg.matrix_exp(matrix_phi)
        matrix_phi_BA = torch.linalg.matrix_exp(-matrix_phi)

        def transform_AB(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    self.torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return multiply_matrix_vectorfield(matrix_phi, coordinates_homogeneous)[
                :, :-1
            ]

        def transform_BA(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [
                    tensor_of_coordinates,
                    torch.ones(shape, device=tensor_of_coordinates.device),
                ],
                axis=1,
            )
            return multiply_matrix_vectorfield(matrix_phi_BA, coordinates_homogeneous)[
                :, :-1
            ]

        return {"phi_AB": transform_AB, "phi_BA": transform_BA}
