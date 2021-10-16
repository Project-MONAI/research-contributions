# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from networks.mlp import MLPBlock
from networks.selfattention import SABlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def loadFrom(self, weights, n_block):
        ROOT = f"module.transformer.blocks.{n_block}."
        block_names = ['mlp.linear1.weight', 'mlp.linear1.bias', 'mlp.linear2.weight', 'mlp.linear2.bias', 'norm1.weight',\
                    'norm1.bias', 'attn.out_proj.weight', 'attn.out_proj.bias', 'attn.qkv.weight', 'norm2.weight',\
                    'norm2.bias']
        with torch.no_grad():
            self.mlp.linear1.weight.copy_(weights['state_dict'][ROOT+block_names[0]])
            self.mlp.linear1.bias.copy_(weights['state_dict'][ROOT+block_names[1]])
            self.mlp.linear2.weight.copy_(weights['state_dict'][ROOT+block_names[2]])
            self.mlp.linear2.bias.copy_(weights['state_dict'][ROOT+block_names[3]])
            self.norm1.weight.copy_(weights['state_dict'][ROOT+block_names[4]])
            self.norm1.bias.copy_(weights['state_dict'][ROOT+block_names[5]])
            self.attn.out_proj.weight.copy_(weights['state_dict'][ROOT+block_names[6]])
            self.attn.out_proj.bias.copy_(weights['state_dict'][ROOT+block_names[7]])
            self.attn.qkv.weight.copy_(weights['state_dict'][ROOT+block_names[8]])
            self.norm2.weight.copy_(weights['state_dict'][ROOT+block_names[9]])
            self.norm2.bias.copy_(weights['state_dict'][ROOT+block_names[10]])


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
