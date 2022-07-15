from transformers.file_utils import (
    ModelOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.utils import logging

from .configuration_swinunetr import SwinUnetrConfig
import torch
from torch import nn
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.utils import BlendMode

import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "darragh/swinunetr-btcv-tiny"
_CONFIG_FOR_DOC = "swinunetrConfig"

SWINUNETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "swinunetr-btcv-tiny",
    "swinunetr-btcv-small",
]

class SwinUnetrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SwinUnetrConfig
    base_model_prefix = "swinunetr"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SwinUnetrModelForInference(SwinUnetrPreTrainedModel):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    Source : https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        
        self.config = config
        
        self.model = SwinUNETR(
            img_size= config.img_size,
            in_channels= config.in_channels,
            out_channels=config. out_channels,
            depths= config.depths,
            num_heads=  config.num_heads,
            feature_size= config.feature_size,
            norm_name= config.norm_name,
            drop_rate= config.drop_rate,
            attn_drop_rate= config.attn_drop_rate,
            dropout_path_rate= config.dropout_path_rate,
            normalize= config.normalize,
            use_checkpoint= config.use_checkpoint,
            spatial_dims= config.spatial_dims,
            )
        
        self.init_weights()
        
    def forward(
        self,
        inputs: torch.Tensor,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT
        ):
        r"""
        Sliding window inference on `inputs` with `predictor`.
    
        The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
        Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
        e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
        could be ([128,64,256], [64,32,128]).
        In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
        an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
        so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).
    
        When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
        To maintain the same spatial sizes, the output image will be cropped to the original input size.
    
        Args:
            inputs: input image to be processed (assuming NCHW[D])
            roi_size: the spatial window size for inferences.
                When its components have None or non-positives, the corresponding inputs dimension will be used.
                if the components of the `roi_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            sw_batch_size: the batch size to run window slices.
            overlap: Amount of overlap between scans.
            mode: {``"constant"``, ``"gaussian"``}
                How to blend output of overlapping windows. Defaults to ``"constant"``.
    
                - ``"constant``": gives equal weight to all predictions.
                - ``"gaussian``": gives less weight to predictions on edges of windows.
            kwargs: optional keyword args to be passed to ``predictor``.
    
        Note:
            - input must be channel-first and have a batch dim, supports N-D sliding window.
    
        """
        
        logits = sliding_window_inference(inputs,
                                          roi_size,
                                          sw_batch_size,
                                          self.model,
                                          overlap,
                                          mode)
        
        return ModelOutput(logits = logits)
        
    
    
        
        
