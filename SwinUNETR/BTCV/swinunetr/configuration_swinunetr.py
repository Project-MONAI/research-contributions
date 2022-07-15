# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Swin Unnetr configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

SWINUNETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "swinunetr-btcv-tiny": "https://huggingface.co/darragh/swinunetr-btcv-tiny/raw/main/config.json",
    "swinunetr-btcv-small": "https://huggingface.co/darragh/swinunetr-btcv-small/raw/main/config.json",
}


class SwinUnetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.

    Examples::

        >>> TBD
    """
    model_type = "swinunetr"

    def __init__(
        self,
        architecture= "SwinUNETR",
        img_size= 96,
        in_channels= 1,
        out_channels= 14,
        depths= (2, 2, 2, 2),
        num_heads=  (3, 6, 12, 24),
        feature_size= 12,
        norm_name= "instance",
        drop_rate= 0.0,
        attn_drop_rate= 0.0,
        dropout_path_rate=  0.0,
        normalize= True,
        use_checkpoint= False,
        spatial_dims= 3,
        **kwargs
    ):
        super().__init__(
            
            architecture= architecture,
            img_size= img_size,
            in_channels= in_channels,
            out_channels= out_channels,
            depths= depths,
            num_heads=  num_heads,
            feature_size= feature_size,
            norm_name= norm_name,
            drop_rate= drop_rate,
            attn_drop_rate= attn_drop_rate,
            dropout_path_rate= dropout_path_rate,
            normalize= normalize,
            use_checkpoint= use_checkpoint,
            spatial_dims= spatial_dims,
            **kwargs,
        )