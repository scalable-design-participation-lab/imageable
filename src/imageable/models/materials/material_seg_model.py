#@markdown Utils (segformer encoder)

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
import os
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()

        use_pretrain = True

        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        if use_pretrain:
            print("loading pretrained model")
            self._load_pretrained_model()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('loaded')

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # for segmentational pre-trained weights
    """
    def _load_pretrained_model(self):
        path = 'modeling/segformer/'
        pretrain_dict = torch.load( path + 'segformer.b2.512x512.ade.160k.pth') # changeable
        pretrain_dict = pretrain_dict['state_dict'] # names are listed as like 'backbone.patch_embed1.proj.weight'
        model_dict = {}
        state_dict = self.state_dict()
        count = 0
        for k, v in pretrain_dict.items():
            name = k.split('backbone.')[-1] # names are listed as like 'patch_embed1.proj.weight'
            if name in state_dict:
                model_dict[name] = v
                print('loaded_' + np.str(count))
                count = count + 1
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print('loading completed')
        """

    # for ImageNet pre-trained weights
    def _load_pretrained_model(self):
        # path = 'weights/init/'
        pretrain_dict = torch.load(str(mit_b2_path)) # changeable
        pretrain_dict = pretrain_dict
        model_dict = {}
        state_dict = self.state_dict()
        count = 0
        for k, v in pretrain_dict.items():
            name = k
            if name in state_dict:
                model_dict[name] = v
                # print('loaded_' + np.str(count))
                count = count + 1
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print('loading completed')


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x




class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


#@markdown Utils (SAMixerHead)
from __future__ import absolute_import

import warnings
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.init import calculate_gain
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.hub import load_state_dict_from_url

import torch.linalg as linalg

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.helpers import build_model_with_cfg

import math
import numpy as np



############# for material encoding module #############
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


##########
class VecAxTrans(nn.Module):
    def __init__(self, mod2dTo1d=True):
        super(VecAxTrans, self).__init__()
        self.mod2dTo1d = mod2dTo1d

    def forward(self, x):

        if self.mod2dTo1d:
            x = x.squeeze(-1).transpose(-1, -2) # 2-D to 1-D for spatial extend

        else:
            x = x.transpose(-1, -2).unsqueeze(-1) # 1-D to 2-D for spatial extend

        return x



class ParamGNorm(nn.Module):
    def __init__(self, H=1, W=1, AxTrans=False, param=True, w=1., k=0.):
        super(ParamGNorm, self).__init__()
        if param:
            if AxTrans:
                points = int(H*W)
                self.gamma = Parameter(w*torch.ones(1, points, 1))
                self.beta = Parameter(k*torch.ones(1, points, 1))
            else:
                self.gamma = Parameter(w*torch.ones(1, 1, H, W))
                self.beta = Parameter(k*torch.ones(1, 1, H, W))

        self.AxTrans = AxTrans
        self.param = param

    def forward(self, x):
        if self.param:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta

            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
        else:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)

            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)

        return x



class LayerGNorm(nn.Module):
    def __init__(self, dim=1, AxTrans=False, param=True, w=1., k=0.):
        super(LayerGNorm, self).__init__()
        if param:
            if AxTrans:
                self.gamma = Parameter(w*torch.ones(1, 1, dim)) # [b,l,c]
                self.beta = Parameter(k*torch.ones(1, 1, dim))
            else:
                self.gamma = Parameter(w*torch.ones(1, dim, 1, 1)) # [b,c,h,w]
                self.beta = Parameter(k*torch.ones(1, dim, 1, 1))

        self.AxTrans = AxTrans
        self.param = param

    def forward(self, x):
        if self.param:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta

            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
                x = self.gamma * x + self.beta
        else:
            if self.AxTrans:
                x = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-12)

            else:
                x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)

        return x



class LayerGNormMS(nn.Module):
    def __init__(self, dim=1, param=True, w=1., k=0., l=6):
        super(LayerGNormMS, self).__init__()
        if param:
            self.gamma = Parameter(w*torch.ones(1, l, dim, 1, 1)) # [b,l,c,h,w]
            self.beta = Parameter(k*torch.ones(1, l, dim, 1, 1))

        self.param = param

    def forward(self, x):
        if self.param:
            x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)
            x = self.gamma * x + self.beta
        else:
            x = (x - x.mean(dim=1, keepdim=True))/(x.std(dim=1, keepdim=True) + 1e-12)

        return x



class ChOverlapAvg(nn.Module):
    def __init__(self, kernel_size=32, reduct_rate=16):
        super(ChOverlapAvg, self).__init__()
        self.pad = nn.ReflectionPad1d(padding=(kernel_size//2, 0)) # left padding only
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=reduct_rate)

        self.kernel_size = kernel_size
        self.reduct_rate = reduct_rate

    def forward(self, x):
        b,c,l = x.shape

        if l < self.kernel_size:
            x = F.avg_pool1d(l)(x)
        else:
            if self.kernel_size > self.reduct_rate:
                x = self.avg(self.pad(x))
            else:
                x = self.avg(x)

        return x

##############




# This module is adopted from Decoupled Dynamic Filter Networks: https://github.com/theFoxofSky/ddfnet (CVPR 2021)
class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel')
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.view(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-12)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b, h, w = x.size(0), x.size(2), x.size(3)
            #l = int(h*w)
            c = self.in_channels
            x = x.view(b, c, -1)
            x = x - x.mean(dim=2, keepdim=True)
            x = x / (x.std(dim=2, keepdim=True) + 1e-12)

            if self.runing_std:
                x = x * self.std[None, :, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None]
            x = x.view(b,c,h,w)
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x
#####################






# SAMixer's definition
class SAMixer(nn.Module):
    def __init__(self, dim=128, spatial=128, rate_reduct=1, branches=4, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super(SAMixer, self).__init__()

        self.spatial = spatial

        self.sign = 'SAMixer' #

        # fusion computation
        group_channels = 64
        self.g = dim // group_channels
        self.d = dim // self.g

        T = 2
        self.T = T
        Ws = T #int(T*T)
        self.paths = branches + 1
        self.avg_local1 = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=Ws, stride=Ws, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True)
                                     ) # sr conv like avg enco

        self.avg_local2 = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=Ws, stride=Ws, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True)
                                     ) # sr conv like avg enco


        self.norm0 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm3 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)
        self.norm4 = nn.LayerNorm(dim, eps=1e-12) #LayerGNorm(dim=dim, AxTrans=True)


        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        self.scl = (self.d)**-0.5

        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim, bias=True)

        self.pos_deco = nn.Sequential(
                                    FilterNorm(dim, 1, 'channel', 'relu', running_std=True),
                                    nn.Conv2d(dim, int(T*T), kernel_size=1, bias=True)
                                    ) # param list: 'in_channels' (heads), 'kernel_size' (win_size), 'type', 'nonlinearity'


        # form-1: 1*1 dense -> 3*3 dw
        self.norm_inner = LayerGNorm(dim=dim) # 2d norm

        expand_ratio = 2
        self.mlp = nn.Sequential(
                                 nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                                 nn.GroupNorm(num_groups=dim, num_channels=dim, eps=1e-12),
                                 nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, bias=True),
                                 act_layer(),
                                 nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, bias=True)
                                     )

        self.norm_outer = nn.LayerNorm(dim, eps=1e-12)

        self.proj_top = nn.Sequential(
                                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                                norm_layer(dim, eps=1e-12),
                                act_layer()
                                         )

        # init
        self.reset_parameters()


    @torch.jit.script
    def combine_mul(x, attn):
        return x * attn

    @torch.jit.script
    def combine_add(x, attn):
        return x + attn

    @torch.jit.script
    def combine_add_inputs(x1, x2, x3, x4):
        return F.interpolate( F.interpolate(F.interpolate(x1, scale_factor=2., mode='bilinear', align_corners=False) \
                                            + x2, scale_factor=2., mode='bilinear', align_corners=False) + x3, scale_factor=2., mode='bilinear', align_corners=False) + x4


    def forward(self, x1, x2, x3, x4):
        b, c, h, w = x1.shape

        ## parametric adaptation (shift computation)
        T, g, d = self.T, self.g, c // self.g

        x = self.combine_add_inputs(x1, x2, x3, x4) # main feature, [b, c, h*8, w*8]

        ## avg enco
        # partition windows
        H, W = h*4, w*4
        N = H * W
        B = b*N

        # MHSA
        box0 = self.norm0(self.avg_local1(x).view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))
        q = self.q(box0).view(B, 1, g, d).transpose(1, 2) * self.scl # only x serves as the q, [B, g, 1, d]
        kv0 = self.kv(box0).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]

        kv1 = F.interpolate(self.kv(self.norm1(x1.view(b, c, N//16).transpose(1, -1).contiguous())).transpose(1, -1).reshape(b, 2*c, h, w), scale_factor=4)\
            .view(b, 1, 2*c, N).permute(0, 3, 1, 2).reshape(B, 1, 2, g, d).permute(2, 0, 3, 1, 4)

        kv2 = F.interpolate(self.kv(self.norm2(x2.view(b, c, N//4).transpose(1, -1).contiguous())).transpose(1, -1).reshape(b, 2*c, h*2, w*2), scale_factor=2)\
            .view(b, 1, 2*c, N).permute(0, 3, 1, 2).reshape(B, 1, 2, g, d).permute(2, 0, 3, 1, 4)

        kv3 = self.kv(self.norm3(x3.view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]

        kv4 = self.kv(self.norm4(self.avg_local2(x4).view(b, 1, c, N).permute(0, 3, 1, 2).reshape(B, 1, c))).view(B, 1, 2, g, d).permute(2, 0, 3, 1, 4) # [2, B, g, 1, d]


        kv = torch.cat([kv0, kv1, kv2, kv3, kv4], dim=3) # [2, B, g, l, d]
        k, v = kv[0], kv[1] # [B, g, l, d]


        sim = q @ k.transpose(-2, -1) # b g 1 d, b g l c -> b g 1 l
        sim = self.softmax(sim)
        u_attn = (sim @ v).transpose(1, 2).reshape(B, 1, c) # b g i j, b g j c -> b g i c (i=1, j=4)

        # position decoding
        u_attn = self.proj(u_attn).view(b, H, W, c).permute(0, 3, 1, 2) # [b, c, h, w]


        u_attn = self.combine_add(u_attn.unsqueeze(2), self.pos_deco(u_attn).unsqueeze(1) )\
            .view(b, c, T, T, H, W).permute(0, 1, 4, 2, 5, 3).reshape(b, c, H*T, W*T) # [b, c, 1, h, w], [b, 1, T*T, h, w], [b, c, T*T, h, w]


        # FFN
        x = self.combine_add(x, u_attn)
        x = self.proj_top(self.combine_add(x, self.norm_outer(self.mlp(self.norm_inner(x)).flatten(2).transpose(1,-1) ).transpose(1,-1).reshape(b, c, H*T, W*T) ) )

        return x

    ############################
    def reset_parameters(self):
        # conv and bn init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
#################





class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

        # init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        # print('decoder init')


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('loaded')

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x




class SAMixerHead(nn.Module):
    """
    """
    def __init__(self, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                 embedding_dim=768, norm_layer=nn.BatchNorm2d, num_classes=20,
                 in_index=[0, 1, 2, 3], dropout_ratio=0.1, input_transform='multiple_select', align_corners=False):
        super(SAMixerHead, self).__init__()
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.mixer = SAMixer(dim=embedding_dim, rate_reduct=1, branches=4, norm_layer=norm_layer)

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # init
        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

        # print('decoder init')



    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print('loaded')



    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs



    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.mixer(_c4, _c3, _c2, _c1)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

#################



########################## ops functions ##########################

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)

#@markdown Utils (SynchronizedBatchNorm2d)
import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

import queue
import collections
import threading

# from .comm import SyncMaster


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.
        Args:
            identifier: an identifier, usually is the device id.
        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).
        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
        Returns: the message to be sent back to the master device.
        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

__all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)
        
        
    
class RMSNet(nn.Module):
    """
    'encoder_id' denotes the type of Mix-Transformer, default: 2, means mit_b2
    """
    def __init__(self, num_classes=20, backbone='mit_b2', encoder_id=2,
                 sync_bn=True, freeze_bn=False):
        super(RMSNet, self).__init__()

        if sync_bn == True:
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self.backbone = mit_b2()
        self.decoder = SAMixerHead(in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                 embedding_dim=768, norm_layer=norm_layer, num_classes=20)

        self.freeze_bn = freeze_bn

    # if there is no se_loss
    def forward(self, input):
        print("Running RMSNet...")
        features = self.backbone(input)
        # print("Built backbone")
        seg_mask = self.decoder(features)
        # print("Built decoder")

        seg_mask = F.interpolate(seg_mask, size=input.size()[2:], mode='bilinear', align_corners=False)

        return seg_mask


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()