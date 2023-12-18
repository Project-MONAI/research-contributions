from collections import namedtuple

import matplotlib
import torch
import torch.nn.functional as F

from icon_registration import config, network_wrappers

from .mermaidlite import compute_warped_image_multiNC


def to_floats(stats):
    out = []
    for v in stats:
        if isinstance(v, torch.Tensor):
            v = torch.mean(v).cpu().item()
        out.append(v)
    return ICONLoss(*out)

class ICON(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def __call__(self, image_A, image_B) -> ICONLoss:
        return super().__call__(image_A, image_B)

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(image_A.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )


class GradICON(network_wrappers.RegistrationModule):
    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )

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

        inverse_consistency_loss = sum(direction_losses)

        return inverse_consistency_loss

    def compute_similarity_measure(self, phi_AB, phi_BA, image_A, image_B):
        self.phi_AB_vectorfield = phi_AB(self.identity_map)
        self.phi_BA_vectorfield = phi_BA(self.identity_map)

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A
        )(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B
        )(self.phi_BA_vectorfield)
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        similarity_loss = self.compute_similarity_measure(
            self.phi_AB, self.phi_BA, image_A, image_B
        )

        inverse_consistency_loss = self.compute_gradient_icon_loss(
            self.phi_AB, self.phi_BA
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )
    

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )

    
    
class BendingEnergy(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):
        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        # dxdx = [f[x+h, y] + f[x-h, y] - 2 * f[x, y]]/(h**2)
        # dxdy = [f[x+h, y+h] + f[x-h, y-h] - f[x+h, y-h] - f[x-h, y+h]]/(4*h**2)
        # BE_2d = |dxdx| + |dydy| + 2 * |dxdy|
        # psudo code: BE_2d = [torch.mean(dxdx**2) + torch.mean(dydy**2) + 2 * torch.mean(dxdy**2)]/4.0  
        # BE_3d = |dxdx| + |dydy| + |dzdz| + 2 * |dxdy| + 2 * |dydz| + 2 * |dxdz|
        
        if len(self.identity_map.shape) == 3:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            bending_energy = torch.mean((dxdx)**2)
            
        elif len(self.identity_map.shape) == 4:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            dydy = (phi_AB_vectorfield[:, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]) / self.spacing[1]**2
            dxdy = (phi_AB_vectorfield[:, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]) / (4.0*self.spacing[0]*self.spacing[1])
            bending_energy = (torch.mean(dxdx**2) + torch.mean(dydy**2) + 2*torch.mean(dxdy**2)) / 4.0
        elif len(self.identity_map.shape) == 5:
            dxdx = (phi_AB_vectorfield[:, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, 1:-1]
                + phi_AB_vectorfield[:, :, :-2]) / self.spacing[0]**2
            dydy = (phi_AB_vectorfield[:, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :-2]) / self.spacing[1]**2
            dzdz = (phi_AB_vectorfield[:, :, :, :, 2:] 
                - 2*phi_AB_vectorfield[:, :, :, :, 1:-1]
                + phi_AB_vectorfield[:, :, :, :, :-2]) / self.spacing[2]**2
            dxdy = (phi_AB_vectorfield[:, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :-2, 2:]) / (4.0*self.spacing[0]*self.spacing[1])
            dydz = (phi_AB_vectorfield[:, :, :, 2:, 2:] 
                + phi_AB_vectorfield[:, :, :, :-2, :-2] 
                - phi_AB_vectorfield[:, :, :, 2:, :-2]
                - phi_AB_vectorfield[:, :, :, :-2, 2:]) / (4.0*self.spacing[1]*self.spacing[2])
            dxdz = (phi_AB_vectorfield[:, :, 2:, :, 2:] 
                + phi_AB_vectorfield[:, :, :-2, :, :-2] 
                - phi_AB_vectorfield[:, :, 2:, :, :-2]
                - phi_AB_vectorfield[:, :, :-2, :, 2:]) / (4.0*self.spacing[0]*self.spacing[2])

            bending_energy = ((dxdx**2).mean() + (dydy**2).mean() + (dzdz**2).mean() 
                    + 2.*(dxdy**2).mean() + 2.*(dydz**2).mean() + 2.*(dxdz**2).mean()) / 9.0
        

        return bending_energy

    def compute_similarity_measure(self, phi_AB_vectorfield, image_A, image_B):

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A
        )(phi_AB_vectorfield)
        
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        )
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        
        similarity_loss = 2 * self.compute_similarity_measure(
            self.phi_AB_vectorfield, image_A, image_B
        )

        bending_energy_loss = self.compute_bending_energy_loss(
            self.phi_AB_vectorfield
        )

        all_loss = self.lmbda * bending_energy_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return BendingLoss(
            all_loss,
            bending_energy_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield),
        )

    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)

class Diffusion(BendingEnergyNet):
    def compute_bending_energy_loss(self, phi_AB_vectorfield):
        phi_AB_vectorfield = self.identity_map - phi_AB_vectorfield
        if len(self.identity_map.shape) == 3:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, 1:-1]
            )**2)

        elif len(self.identity_map.shape) == 4:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2)
        elif len(self.identity_map.shape) == 5:
            bending_energy = torch.mean((
                - phi_AB_vectorfield[:, :, 1:]
                + phi_AB_vectorfield[:, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :-1]
            )**2) + torch.mean((
                - phi_AB_vectorfield[:, :, :, :, 1:]
                + phi_AB_vectorfield[:, :, :, :, :-1]
            )**2)


        return bending_energy * self.identity_map.shape[2] **2

