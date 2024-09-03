import torch
from torch import nn
import math
import torch.nn.functional as F

from torchvision.ops.misc import interpolate

from utils import make_divisible

class ConvLayer2d(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 dilation=1,
                 use_act=True, use_norm=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size // 2),
                              groups=groups,
                              dilation=dilation,
                              bias=True)
        if use_act:
            self.act = nn.ReLU()
        else:
            self.act = None

        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MVBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 skip_connection=True):
        super().__init__()

        assert stride in [1, 2]

        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        self.block= nn.Sequential(
            ConvLayer2d(in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=1),

            ConvLayer2d(in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=stride,
                        groups=hidden_dim,
                        dilation=dilation),
            ConvLayer2d(in_channels=hidden_dim,
                        out_channels=out_channels,
                        kernel_size=1,
                        use_act=False)
                                  )

        self.use_res_connect = stride == 1 and in_channels == out_channels and skip_connection


    def forward(self, x):
        if self.use_res_connect:
            y = x + self.block(x)
        else:
            y = self.block(x)
        return y


def unfolding(feature_map, patch_size):
    patch_h, patch_w = patch_size[0], patch_size[1]

    B, C, H, W = feature_map.shape

    new_H = int(math.ceil(H / patch_h) * patch_h)
    new_W = int(math.ceil(W / patch_w) * patch_w)

    interpolate = False

    if new_H != H or new_W != W:
        feature_map = F.interpolate(
            feature_map, size=(new_H, new_W), mode="bilinear", align_corners=False
        )
        interpolate = True

    n_patch_w = new_W // patch_w
    n_patch_h = new_H // patch_h

    n_patch = n_patch_w * n_patch_h

    reshape_fm = feature_map.view(B, C, patch_h, n_patch_h, patch_w, n_patch_w)  # (B, C, H, W) => (B, C, ph, nh, pw, nw)
    reshape_fm = reshape_fm.permute(0, 1, 2, 4, 3, 5)  # (B, C, ph, nh, pw, nw) => (B, C, ph, pw, nh, nw)
    reshape_fm = reshape_fm.reshape(B, C, patch_h * patch_w, n_patch_h * n_patch_w)  # (B, C, ph, pw, nh, nw) => (B, C, P, N)
    reshape_fm = reshape_fm.permute(0, 2, 3, 1)  # (B, C, P, N) => (B, P, N, C)
    reshape_fm = reshape_fm.reshape(-1, n_patch, C)

    info_dict = {
        "orig_size": (H, W),
        "batch_size": B,
        "interpolate": interpolate,
        "num_patches_w": n_patch_w,
        "num_patches_h": n_patch_h,
        "patch_w": patch_w,
        "patch_h": patch_h
    }
    return reshape_fm, info_dict


def folding(patches, info_dict):
    n_dim = patches.dim()
    assert n_dim == 3, f"Tensor should be of shape BPxNxC. Got: {patches.shape}"

    B1, N, C = patches.shape

    B = info_dict['batch_size']
    n_patch_w = info_dict['num_patches_w']
    n_patch_h = info_dict['num_patches_h']
    n_patch = n_patch_w * n_patch_h
    H, W = info_dict['orig_size']

    patch_w = info_dict['patch_w']
    patch_h = info_dict['patch_h']

    patches = patches.contiguous().view(B, patch_w * patch_h, n_patch, -1)  # (B, P, N, C)

    feature_map = patches.permute(0, 3, 2, 1)  # (B, P, N, C) => (B, C, P, N)
    feature_map = feature_map.view(B, C, patch_h, patch_w, N)  # (B, C, P, N) => (B, C, p_h. p_w, N)

    feature_map = feature_map.view(B, C, patch_h, patch_w, n_patch_h, n_patch_w)  # (B, C, p_h. p_w, N) => (B, C, p_h. p_w, n_h, n_w)
    feature_map = feature_map.permute(0, 1, 2, 4, 3, 5)  # (B, C, p_h. p_w, n_h, n_w) => (B, C, p_h, n_h, p_w, n_w)
    feature_map = feature_map.reshape(B, C, patch_h * n_patch_h, patch_w * n_patch_w)

    if info_dict['interpolate']:
        feature_map = F.interpolate(
            feature_map,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
    return feature_map


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, drop_rate=0.5):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        y = self.mlp(x)
        return y


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_head=6, head_dim=64, drop_rate=0.5):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(input_dim)
        self.qs = nn.ModuleList([nn.Linear(input_dim, head_dim) for _ in range(num_head)])
        self.ks = nn.ModuleList([nn.Linear(input_dim, head_dim) for _ in range(num_head)])
        self.vs = nn.ModuleList([nn.Linear(input_dim, head_dim) for _ in range(num_head)])
        self.projection = nn.Sequential(
            nn.Linear(head_dim * num_head, input_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.scale = head_dim ** -0.5

    def forward(self, x):  # x shape: (B, N, C) B: batch_size, N: number patch, C: embedding dim
        attn_outs = []
        y = self.norm(x)
        for i in range(self.num_head):
            q = self.qs[i](y)
            k = self.ks[i](y)
            v = self.vs[i](y)

            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

            attn_out = torch.matmul(attn, v)
            attn_outs.append(attn_out)

        y = torch.cat(attn_outs, dim=-1)
        y = self.projection(y)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, num_head=6, head_dim=64, hidden_dim=64,
                 attn_drop_rate=0.5, drop_rate=0.5
                 ):
        super(TransformerEncoder, self).__init__()
        self.attentions = MultiHeadSelfAttention(input_dim, num_head, head_dim, attn_drop_rate)
        self.mlp = MLP(input_dim, hidden_dim, out_dim, drop_rate)

    def forward(self, x):

        y = self.attentions(x)
        x = x + y
        y = self.mlp(x)
        y = x + y
        return y
