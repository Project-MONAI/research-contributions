import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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

    def as_function(self, image):
        """image is a tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of intensities.
        """

        return lambda coordinates: compute_warped_image_multiNC(
            image, coordinates, self.spacing, 1
        )

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        # if parents_identity_map is not None:
        #    self.identity_map = parents_identity_map
        # else:
        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id), persistent=False)

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
        field_0_1 = transform(identity_map) - self.identity_map
        network_shape_list = list(self.identity_map.shape[2:])
        scale = torch.Tensor(network_shape_list).to(self.identity_map.device)

        for _ in network_shape_list:
            scale = scale[:, None]
        scale = scale[None, :]
        field_spacing_1 = scale * field_0_1
        return field_spacing_1
    def make_ddf_using_icon_module(self, image_A, image_B):
        """Compute a deformation field compatible with monai's Warp 
        using an ICON RegistrationModule. If the RegistrationModule returns a transform, this function
        returns the monai version of that transform. If the RegistrationModule returns a loss,
        this function returns a monai version of the internal transform as well as the loss.
        """

        res = self(image_A, image_B)
        field = self.make_ddf_from_icon_transform(res["phi_AB"]
                                                  )
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


