import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.special import logit

from blocks import *


class MobileViT(nn.Module):
    def __init__(self, opts):
        super().__init__()

        num_classes = opts['num_classes']
        img_channels = opts['img_channels']
        dim_in = opts['dim_in']

        self.init_conv = ConvLayer2d(in_channels=img_channels, out_channels=dim_in, kernel_size=3, stride=2)

        config = opts['config']

        last_exp_factor = config.pop('last_exp_factor')

        self.blocks = nn.ModuleList([])
        for layer, layer_cfg in config.items():
            block, dim_out = self._make_layers(dim_in, layer_cfg)

            dim_in = dim_out
            self.blocks.append(block)

        last_dim = dim_in * last_exp_factor

        self.head = nn.Sequential(
            ConvLayer2d(in_channels=dim_in,
                        out_channels=max(last_dim, dim_in),
                        kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = MLP(input_dim=last_dim, hidden_dim=(last_dim // 2), out_dim=num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        x = x.view(B, -1)
        logits = self.classifier(x)
        return logits

    @staticmethod
    def _make_layers(dim_in, cfg):
        block = []
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            stride = cfg.get('stride', 1)
            if stride == 2:

                layer = MVBlock(in_channels=dim_in,
                                out_channels=cfg.get("out_channels"),
                                stride=stride,
                                expand_ratio=cfg.get("mv_expand_ratio", 4),
                                dilation=1
                                )
                block.append(layer)
                dim_in = cfg.get("out_channels")

            head_dim = cfg.get("head_dim", 32)
            transformer_dim = cfg['transformer_channels']
            ffn_dim = cfg.get('ffn_dim')

            if head_dim is None:
                num_heads = cfg.get("num_heads", 4)
                if num_heads is None:
                    num_heads = 4
                head_dim = transformer_dim // num_heads

            block.append(
                MobileViTBlock(
                    input_dim=dim_in,
                    transformer_dim=transformer_dim,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    attn_dropout=cfg.get("attn_dropout", 0.0),
                    ffn_dropout=cfg.get("ffn_dropout", 0.0),
                    num_transformer=cfg.get('n_transformer', 1),
                    patch_size=cfg.get("patch_size", (2, 2)),
                )
            )

        else:
            out_channels = cfg.get("out_channels")
            num_blocks = cfg.get("num_blocks", 2)
            expand_ratio = cfg.get("expand_ratio", 4)

            for i in range(num_blocks):
                stride = cfg.get('stride', 1) if i == 0 else 1

                layer = MVBlock(in_channels=dim_in,
                                out_channels=out_channels,
                                stride=stride,
                                expand_ratio=expand_ratio)
                block.append(layer)
                dim_in = out_channels

        return nn.Sequential(*block), dim_in


class MobileViTBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 transformer_dim,
                 head_dim=32,
                 ffn_dim=128,
                 attn_dropout=0.0,
                 ffn_dropout=0.0,
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

        num_heads = transformer_dim // head_dim

        self.global_rep = nn.Sequential(*[
                TransformerEncoder(
                    transformer_dim, transformer_dim,
                    num_head=num_heads, head_dim=head_dim, hidden_dim=ffn_dim,
                    attn_drop_rate=attn_dropout, drop_rate=ffn_dropout
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


if __name__ == "__main__":
    from mobilevit_cfg import opts

    model = MobileViT(opts)
    model = model.cuda()

    x = torch.rand(64, 3, 256, 256).cuda()

    y = model(x)
    print(f"y shape: {y.shape}")
