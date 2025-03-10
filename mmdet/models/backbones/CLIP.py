import torch.nn as nn
import open_clip
import torch
from ..builder import BACKBONES
import time
import torch.nn.functional as F
import logging
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import numpy as np
from mmcv.utils import get_logger
# from ..utils import PatchEmbed



def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=np.float64, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        print(f'embedding_size: {embedding_size}')
        try:
            num_patches = model.patch_embed.num_patches
            print(f'num_patches: {num_patches}')
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(orig_size, new_size)
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    elif 'visual.positional_embedding' in checkpoint_model:
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        pos_embed_checkpoint = checkpoint_model['visual.positional_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # print(f'embedding_size: {embedding_size}')
        # print(f'model.visual:\n{model.visual}')

        image_size=model.visual.image_size[0]
        patch_size=model.visual.patch_size[0]
        num_patches = (image_size // patch_size) ** 2
        num_extra_tokens=1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(orig_size, new_size)
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            # print(f'pos_embed_checkpoint.shape: {pos_embed_checkpoint.shape}') #pos_embed_checkpoint.shape: torch.Size([50, 768])
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
            # print(f'extra_tokens.shape: {extra_tokens.shape}') #extra_tokens.shape: torch.Size([1, 768])
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
            # print(f'pos_tokens.shape: {pos_tokens.shape}') pos_tokens.shape: torch.Size([49, 768])
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens.squeeze(0)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            checkpoint_model['visual.positional_embedding'] = new_pos_embed
    else:
        print(f'error interplot pos embed')
        print(model.visual)


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

@BACKBONES.register_module()
class CLIP():
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,pretrained=None,model_name=None, **kwargs):

        self.pretrained = pretrained
        clip_model = open_clip.create_model_and_transforms(model_name)
        self.model = clip_model[0]

        embed_dim = self.model.visual.width
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # checkpoint = torch.load(pretrained)
        # interpolate_pos_embed(self.model, checkpoint)
        #
        # msg = self.model.load_state_dict(checkpoint, strict=False)
        # print(f'load state dict message:\n{msg}')
        #
        device = torch.device('cuda:0')
        self.model = self.model.to(device)
        self.fpn1 = self.fpn1.to(device)
        self.fpn2 = self.fpn2.to(device)
        self.fpn3 = self.fpn3.to(device)
        self.fpn4 = self.fpn4.to(device)


    def __call__(self, x):
        self.forward(x)

    # def backbone(self,x):
    #     self.forward(x)

    def init_weights(self, pretrained=None):

        if pretrained is None:
            pretrained = self.pretrained
        else:
            print(f'Error loading pretrained weights from {pretrained}')
        checkpoint = torch.load(pretrained)
        # for name in checkpoint.keys():
        #     print(name)
        # for name, param in checkpoint.items():
        #     print(f'Name: {name}, Shape: {param.shape}')
        interpolate_pos_embed(self.model, checkpoint)

        msg = self.model.load_state_dict(checkpoint, strict=False)
        print(f'load state dict message:\n{msg}')

        device = torch.device('cuda:0')
        self.model = self.model.to(device)
        self.fpn1 = self.fpn1.to(device)
        self.fpn2 = self.fpn2.to(device)
        self.fpn3 = self.fpn3.to(device)
        self.fpn4 = self.fpn4.to(device)


    def forward_features(self, x):
        x = self.model.forward(x)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            x[i] = ops[i](x[i])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x
