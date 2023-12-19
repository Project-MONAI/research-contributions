from collections import namedtuple

import matplotlib
import torch
import torch.nn.functional as F

from icon_registration import config, network_wrappers

import registration_module


class Loss(registration_module.RegistrationModule):
    def __init__(self, network, similarity, lmbda):
        super().__init__()
        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_similarity(self, image_A, image_B, phi_AB_vectorfield):
        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros(
                [image_A.shape[0]] + [1] + list(image_A.shape[2:]),
                device=image_A.device,
            )
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1)
            if inbounds_tag is not None
            else image_A
        )(
            self.phi_AB_vectorfield,
        )
        similarity_loss = self.similarity(warped_image_A, image_B)
        return {"similarity_loss": similarity_loss, "warped_image_A": warped_image_A}


class TwoWayRegularizer(Loss):
    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        phi_AB = self.regis_net(image_A, image_B)["phi_AB"]
        phi_BA = self.regis_net(image_B, image_A)["phi_AB"]

        phi_AB_vectorfield = phi_AB(self.identity_map)
        phi_BA_vectorfield = phi_BA(self.identity_map)

        similarity_AB = self.compute_similarity(image_A, image_B, phi_AB_vectorfield)
        similarity_BA = self.compute_similarity(image_B, image_A, phi_BA_vectorfield)

        similarity_loss = (
            similarity_AB["similarity_loss"] + similarity_BA["similarity_loss"]
        )
        regularization_loss = compute_regularizer(self, phi_AB, phi_BA)

        all_loss = self.lmbda * gradient_inverse_consistency_loss + similarity_loss

        negative_jacobian_voxels = flips(phi_BA_vectorfield)

        return {
            "all_loss": all_loss,
            "regularization_loss": inverse_consistency_loss,
            "similarity_loss": similarity_loss,
            "phi_AB": phi_AB,
            "phi_BA": phi_BA,
            "warped_image_A": similarity_AB["warped_image_A"],
            "warped_image_B": similiarity_BA["warped_image_A"],
            "negative_jacobian_voxels": negative_jacobian_voxels,
        }


class ICON(TwoWayRegularizer):
    def compute_regularizer(self, phi_AB, phi_BA):
        Iepsilon = self.identity_map + torch.randn(*self.identity_map.shape).to(
            image_A.device
        )

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        inverse_consistency_loss /= self.input_shape[2] ** 2

        return inverse_consistency_loss


class GradICON(TwoWayRegularizer):
    def compute_regularizer(self, phi_AB, phi_BA):
        Iepsilon = self.identity_map + torch.randn(*self.identity_map.shape).to(
            self.identity_map.device
        )
        if len(self.input_shape) - 2 == 3:
            Iepsilon = Iepsilon[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = Iepsilon[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        gradient_inverse_consistency_loss = sum(direction_losses)

        return gradient_inverse_consistency_loss


class OneWayRegularizer(Loss):
    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        phi_AB = self.regis_net(image_A, image_B)["phi_AB"]

        phi_AB_vectorfield = phi_AB(self.identity_map)

        similarity_AB = self.compute_similarity(image_A, image_B, phi_AB_vectorfield)

        similarity_loss = 2 * similarity_AB["similarity_loss"]
        regularization_loss = compute_regularizer(self, phi_AB_vectorfield)

        all_loss = self.lmbda * regularization_loss + similarity_loss

        negative_jacobian_voxels = flips(phi_AB_vectorfield)

        return {
            "all_loss": all_loss,
            "regularization_loss": regularization_loss,
            "similarity_loss": similarity_loss,
            "phi_AB": phi_AB,
            "warped_image_A": similarity_AB["warped_image_A"],
            "negative_jacobian_voxels": negative_jacobian_voxels,
        }


class BendingEnergy(OneWayRegularizer):
    def compute_regularizer(self, phi_AB_vectorfield):
        # dxdx = [f[x+h, y] + f[x-h, y] - 2 * f[x, y]]/(h**2)
        # dxdy = [f[x+h, y+h] + f[x-h, y-h] - f[x+h, y-h] - f[x-h, y+h]]/(4*h**2)
        # BE_2d = |dxdx| + |dydy| + 2 * |dxdy|
        # pseudo code: BE_2d = [torch.mean(dxdx**2) + torch.mean(dydy**2) + 2 * torch.mean(dxdy**2)]/4.0
        # BE_3d = |dxdx| + |dydy| + |dzdz| + 2 * |dxdy| + 2 * |dydz| + 2 * |dxdz|

        if len(self.identity_map.shape) == 3:
            dxdx = (
                phi_AB_vectorfield[:, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]
            ) / self.spacing[0] ** 2
            bending_energy = torch.mean((dxdx) ** 2)

        elif len(self.identity_map.shape) == 4:
            dxdx = (
                phi_AB_vectorfield[:, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]
            ) / self.spacing[0] ** 2
            dydy = (
                phi_AB_vectorfield[:, :, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]
            ) / self.spacing[1] ** 2
            dxdy = (
                phi_AB_vectorfield[:, :, 2:, 2:]
                + phi_AB_vectorfield[:, :, :-2, :-2]
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]
            ) / (4.0 * self.spacing[0] * self.spacing[1])
            bending_energy = (
                torch.mean(dxdx**2)
                + torch.mean(dydy**2)
                + 2 * torch.mean(dxdy**2)
            ) / 4.0
        elif len(self.identity_map.shape) == 5:
            dxdx = (
                phi_AB_vectorfield[:, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]
            ) / self.spacing[0] ** 2
            dydy = (
                phi_AB_vectorfield[:, :, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]
            ) / self.spacing[1] ** 2
            dzdz = (
                phi_AB_vectorfield[:, :, :, :, 2:]
                - 2 * phi_AB_vectorfield[:, :, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :, :-2]
            ) / self.spacing[2] ** 2
            dxdy = (
                phi_AB_vectorfield[:, :, 2:, 2:]
                + phi_AB_vectorfield[:, :, :-2, :-2]
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]
            ) / (4.0 * self.spacing[0] * self.spacing[1])
            dydz = (
                phi_AB_vectorfield[:, :, :, 2:, 2:]
                + phi_AB_vectorfield[:, :, :, :-2, :-2]
                - phi_AB_vectorfield[:, :, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :, :-2, 2:]
            ) / (4.0 * self.spacing[1] * self.spacing[2])
            dxdz = (
                phi_AB_vectorfield[:, :, 2:, :, 2:]
                + phi_AB_vectorfield[:, :, :-2, :, :-2]
                - phi_AB_vectorfield[:, :, 2:, :, :-2]
                - phi_AB_vectorfield[:, :, :-2, :, 2:]
            ) / (4.0 * self.spacing[0] * self.spacing[2])

            bending_energy = (
                (dxdx**2).mean()
                + (dydy**2).mean()
                + (dzdz**2).mean()
                + 2.0 * (dxdy**2).mean()
                + 2.0 * (dydz**2).mean()
                + 2.0 * (dxdz**2).mean()
            ) / 9.0

        return bending_energy


class Diffusion(OneWayRegularizer):
    def compute_regularizer(self, phi_AB_vectorfield):
        phi_AB_vectorfield = self.identity_map - phi_AB_vectorfield
        if len(self.identity_map.shape) == 3:
            bending_energy = torch.mean(
                (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, 1:-1]) ** 2
            )

        elif len(self.identity_map.shape) == 4:
            bending_energy = torch.mean(
                (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, :-1]) ** 2
            ) + torch.mean(
                (-phi_AB_vectorfield[:, :, :, 1:] + phi_AB_vectorfield[:, :, :, :-1])
                ** 2
            )
        elif len(self.identity_map.shape) == 5:
            bending_energy = (
                torch.mean(
                    (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, :-1]) ** 2
                )
                + torch.mean(
                    (
                        -phi_AB_vectorfield[:, :, :, 1:]
                        + phi_AB_vectorfield[:, :, :, :-1]
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        -phi_AB_vectorfield[:, :, :, :, 1:]
                        + phi_AB_vectorfield[:, :, :, :, :-1]
                    )
                    ** 2
                )
            )

        return bending_energy * self.identity_map.shape[2] ** 2


class VelocityFieldDiffusion(Diffusion):
    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        phi_AB_dict = self.regis_net(image_A, image_B)
        phi_AB = phi_AB_dict["phi_AB"]

        phi_AB_vectorfield = phi_AB(self.identity_map)

        similarity_AB = self.compute_similarity(image_A, image_B, phi_AB_vectorfield)

        similarity_loss = 2 * similarity_AB["similarity_loss"]

        velocity_fields = phi_AB["velocity_fields"]
        regularization_loss = 0
        for v in velocity_fields:
            regularization_loss += compute_regularizer(self, phi_AB_vectorfield)

        all_loss = self.lmbda * regularization_loss + similarity_loss

        negative_jacobian_voxels = flips(phi_AB_vectorfield)

        return {
            "all_loss": all_loss,
            "regularization_loss": regularization_loss,
            "similarity_loss": similarity_loss,
            "phi_AB": phi_AB,
            "warped_image_A": similarity_AB["warped_image_A"],
            "negative_jacobian_voxels": negative_jacobian_voxels,
        }


class VelocityFieldBendingEnergy(BendingEnergy):
    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        phi_AB_dict = self.regis_net(image_A, image_B)
        phi_AB = phi_AB_dict["phi_AB"]

        phi_AB_vectorfield = phi_AB(self.identity_map)

        similarity_AB = self.compute_similarity(image_A, image_B, phi_AB_vectorfield)

        similarity_loss = 2 * similarity_AB["similarity_loss"]

        velocity_fields = phi_AB["velocity_fields"]
        regularization_loss = 0
        for v in velocity_fields:
            regularization_loss += compute_regularizer(self, phi_AB_vectorfield)

        all_loss = self.lmbda * regularization_loss + similarity_loss

        negative_jacobian_voxels = flips(phi_AB_vectorfield)

        return {
            "all_loss": all_loss,
            "regularization_loss": regularization_loss,
            "similarity_loss": similarity_loss,
            "phi_AB": phi_AB,
            "warped_image_A": similarity_AB["warped_image_A"],
            "negative_jacobian_voxels": negative_jacobian_voxels,
        }
