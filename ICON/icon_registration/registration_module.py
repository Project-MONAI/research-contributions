import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from monai.networks.blocks import Warp
from monai.networks.utils import meshgrid_ij


class RegistrationModule(nn.Module):
    r"""Base class for icon modules that perform registration.

    A subclass of RegistrationModule should have a forward method that
    takes as input two images image_A and image_B, and returns a python function
    phi_AB that transforms a tensor of coordinates.

    RegistrationModule provides a method as_function that turns a tensor
    representing an image into a python function mapping a tensor of coordinates
    into a tensor of intensities :math:`\mathbb{R}^N \rightarrow \mathbb{R}` .
    Mathematically, this is what an image is anyway.

    After this class is constructed, but before it is used, you _must_ call
    assign_identity_map on it or on one of its parents to define the coordinate
    system associated with input images.

    The contract that a successful registration fulfils is:
    for a tensor of coordinates X, self.as_function(image_A)(phi_AB(X)) ~= self.as_function(image_B)(X)

    ie

    .. math::
        I^A \circ \Phi^{AB} \simeq I^B

    In particular, self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B
    """

    def __init__(self):
        super().__init__()
        self.downscale_factor = 1
        self.warp = Warp()
        self.identity_map = None

    def _make_identity_map(self, shape):
        mesh_points = [torch.arange(0, dim) for dim in shape[2:]]
        grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
        grid = torch.stack([grid], dim=0).float()  # (batch, spatial_dims, ...)
        return grid

    def as_function(self, image):
        """image is a (potentially vector valued) tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of the intensity of `image` at `coordinates`.

        This allows translating the standard notation of registration papers more literally into code.

        I \\circ \\Phi , the standard mathematical notation for a warped image, has the type
        "function from coordinates to intensities" and can be translated to the python code

        warped_image = lambda coords: self.as_function(I)(phi(coords))

        Often, this should actually be left as a function. If a tensor is needed, conversion is:

        warped_image_tensor = warped_image(self.identity_map)
        """

        def partially_applied_warp(coordinates):
            coordinates_shape = list(coordinates.shape)
            coordinates_shape[0] = image.shape[0]
            coordinates = torch.broadcast_to(coordinates, coordinates_shape)
            return self.warp(image, coordinates.clone())

        return partially_applied_warp

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = input_shape
        grid = self._make_identity_map(input_shape)
        del self.identity_map
        self.register_buffer("identity_map", grid, persistent=False)

        if self.downscale_factor != 1:
            child_shape = np.concatenate(
                [
                    self.input_shape[:2],
                    np.ceil(self.input_shape[2:] / self.downscale_factor).astype(int),
                ]
            )
        else:
            child_shape = self.input_shape
        for child in self.children():
            if isinstance(child, RegistrationModule):
                child.assign_identity_map(
                    child_shape,
                    # None if self.downscale_factor != 1 else self.identity_map,
                )

    def make_ddf_from_icon_transform(self, transform):
        """Compute A deformation field compatible with monai's Warp
        using an ICON transform. The assosciated ICON identity_map is also required
        """
        return transform(self.identity_map) - self.identity_map

    def make_ddf_using_icon_module(self, image_A, image_B):
        """Compute a deformation field compatible with monai's Warp
        using an ICON RegistrationModule. If the RegistrationModule returns a transform, this function
        returns the monai version of that transform. If the RegistrationModule returns a loss,
        this function returns a monai version of the internal transform as well as the loss.
        """

        res = self(image_A, image_B)
        field = self.make_ddf_from_icon_transform(res["phi_AB"])
        return field, res

    def forward(image_A, image_B):
        """Register a pair of images:
        return a python function phi_AB that warps a tensor of coordinates such that

        .. code-block:: python

            self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B

        .. math::
            I^A \circ \Phi^{AB} \simeq I^B

        :param image_A: the moving image
        :param image_B: the fixed image
        :return: :math:`\Phi^{AB}`
        """
        raise NotImplementedError()
