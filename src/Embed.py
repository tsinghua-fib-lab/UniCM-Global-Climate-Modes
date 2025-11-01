import numpy as np
import torch.nn as nn
import torch
from torch_geometric.nn.conv import GCNConv
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

class TokenEmbedding_S(nn.Module):
    def __init__(self, c_in, d_model,  patch_size):
        super(TokenEmbedding_S, self).__init__()
        kernel_size = [patch_size, patch_size]
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B*T, C, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(2)
        x = torch.einsum("ncs->nsc", x)  # [N, H*W, C]
        return x

class TokenEmbedding_ST(nn.Module):
    def __init__(self, c_in, d_model, patch_size, t_patch_len,  stride):
        super(TokenEmbedding_ST, self).__init__()
        kernel_size = [t_patch_len, patch_size, patch_size]
        stride_size = [stride, patch_size,  patch_size]
        self.tokenConv = nn.Conv3d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # B, C, T, H, W = x.shape
        x = self.tokenConv(x)
        x = x.flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # [N, T*H*W, C]
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb