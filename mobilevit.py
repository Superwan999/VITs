import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from blocks import *


class MobileViTBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 transformer_dim,
                 head_dim=32,
                 hidden_dim=128,
                 attn_dropout=0.0,
                 mlp_dropout=0.0,
                 num_transformer=2,
                 patch_size=(8, 8),
                 no_fusion=False
                 ):
        super().__init__()

        self.local_rep = nn.Sequential(
            ConvLayer2d(in_channels=input_dim,
                        out_channels=input_dim,
                        kernel_size=3),
            ConvLayer2d(in_channels=input_dim,
                        out_channels=transformer_dim,
                        kernel_size=1,
                        use_act=False, use_norm=False)
        )

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        self.global_rep = nn.Sequential(*[
                TransformerEncoder(
                    transformer_dim, transformer_dim,
                    num_head=num_heads, head_dim=head_dim, hidden_dim=hidden_dim,
                    attn_drop_rate=attn_dropout, drop_rate=mlp_dropout
                ) for _ in range(num_transformer)
        ])

        self.conv_proj = ConvLayer2d(in_channels=transformer_dim,
                                     out_channels=input_dim,
                                     kernel_size=1)

        self.patch_size = patch_size
        self.fusion = None

        if not no_fusion:
            self.fusion = ConvLayer2d(in_channels=2 * input_dim, out_channels=input_dim, kernel_size=3)

    def forward(self, x):
        res = x

        fm = self.local_rep(x)

        patches, info_dict = unfolding(fm, self.patch_size)

        patches = self.global_rep(patches)

        fm = folding(patches, info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm
