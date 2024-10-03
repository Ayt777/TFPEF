import math
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

from timm.models.layers import DropPath, trunc_normal_, activations
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from timm.models.vision_transformer import Block as ViTBlock
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from mmengine.model import  kaiming_init, normal_init

from .layers import (HorBlock, ChannelAggregationFFN, MultiOrderGatedAggregation,
                     PoolFormerBlock, CBlock, SABlock, MixMlp, VANBlock)


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class BasicConv3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv3d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

    

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 seq_len,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 use_tada=False,
                 layer_scale_init_value=1e-6):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)
        # self.conv = BasicConv3d(seq_len, seq_len, kernel_size=kernel_size, stride=stride,
        #                         upsampling=upsampling, padding=padding,
        #                         act_norm=act_norm, act_inplace=act_inplace)
        self.conv_rf = RouteFuncMLP(
            c_in=seq_len,  # number of input filters
            ratio=1,  # reduction ratio for MLP
            kernels=[3, 3],  # list of temporal kernel sizes
        )
        self.conv_tada = TAdaConv2d(
            in_channels=seq_len,
            out_channels=seq_len,
            kernel_size=[1, 3, 3],  # usually the temporal kernel size is fixed to be 1
            stride=[1, 1, 1],  # usually the temporal stride is fixed to be 1
            padding=[0, 1, 1],  # usually the temporal padding is fixed to be 0
            bias=False,
            cal_dim="cin"
        )
        # self.norm = LayerNorm(seq_len, eps=1e-6)
        # self.avgpool = nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        # self.norm_avgpool = LayerNorm(seq_len, eps=1e-6)
        # self.norm_avgpool.weight.data.zero_()
        # self.norm_avgpool.bias.data.zero_()
        # self.pwconv1 = nn.Linear(seq_len, 4 * seq_len)
        # self.act = QuickGELU()
        # self.pwconv2 = nn.Linear(4 * seq_len, seq_len)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((seq_len)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = nn.Identity()
        self.T = seq_len
        self.use_tada = use_tada
    #     self.apply(self._init_weights)
        
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        if self.use_tada:
            # 串联
            if len(x.shape)==4:
                _,c,h,w = x.shape
                x = x.reshape(-1,self.T,c,h,w)
            else:
                b,t,c,h,w = x.shape
                x = x.reshape(b,t,c,h,w)
            # shortcut = x
            x = self.conv_tada(x, self.conv_rf(x))
            # # temporal aggregation
            # norm_avgpool_x = self.avgpool(x)
            # x = x.permute(0, 2, 3, 4, 1)  # (N, C, T, H, W) -> (N, T, H, W, C)
            # norm_avgpool_x = norm_avgpool_x.permute(0, 2, 3, 4, 1)  # (N, C, T, H, W) -> (N, T, H, W, C)
            # x = self.norm(x) + self.norm_avgpool(norm_avgpool_x)

            # x = self.pwconv1(x)
            # x = self.act(x)
            # x = self.pwconv2(x)
            # if self.gamma is not None:
            #     x = self.gamma * x
            # x = x.permute(0, 4, 1, 2, 3)  # (N, T, H, W, C) -> (N, C, T, H, W)
            # y = shortcut + self.drop_path(x)
            
            x = x.reshape(-1,c,h,w)
            y = self.conv(x)
            # 并联
            # if len(x.shape)==4:
            #     x = x.unsqueeze(0)
            # y = self.conv_tada(x, self.conv_rf(x))
            # y = y.squeeze(0)
            # y = self.conv(x.squeeze(0)+y)
        else:
            if len(x.shape)==5:
                b,t,c,h,w = x.shape
                x = x.reshape(-1,c,h,w)
            # if len(x.shape)==4:
            #     bt,c,h,w = x.shape
            #     x = x.reshape(-1,self.T,c,h,w)
            y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""
    # 这个类实现了simvpv2中公式12-14的步骤
    def __init__(self, dim, kernel_size, seq_len , dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        
        self.T = seq_len
        # self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        # self.conv_spatial = nn.Conv2d(
        #     dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        u = x.clone()
        b,c,h,w = x.shape
        # x = x.reshape(b,self.T,-1,h,w)
        # x = x.view(1,5,-1,15,15)
        # x = self.conv(x, self.conv_rf(x))
        # x = x.reshape(1,-1,15,15)
        # attn = self.conv0(x)           # depth-wise conv
        # attn = self.conv_spatial(attn) # depth-wise dilation convolution
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x
    
    

class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, seq_len, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        # self.spatial_gating_unit = SequentialPolarizedSelfAttention_1(T=seq_len,channel=d_model//seq_len)
        # self.spatial_gating_unit = SequentialPolarizedSelfAttention_2(T=seq_len,channel=d_model//seq_len)
        # self.spatial_gating_unit = ParallelPolarizedSelfAttention_1(T=seq_len,channel=d_model//seq_len)
        # self.spatial_gating_unit = AttentionModule(d_model, kernel_size, seq_len)
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        # self.spatial_gating_unit = TemporalAttentionModule_a(d_model, kernel_size)
        # self.spatial_gating_unit = TemporalAttentionModule_1(d_model//seq_len, kernel_size, seq_len)
        # self.spatial_gating_unit = SequentialPolarizedSelfAttention(channel=d_model)
        # self.spatial_gating_unit = DAModule(d_model=d_model,kernel_size=3,H=15,W=15)
        # self.spatial_gating_unit = CoordAtt(d_model, d_model, reduction=32)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # print(x.shape)
        x = self.spatial_gating_unit(x)
        # print(x.shape)
        # 四维输入输出 形状一致
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x

class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, seq_len, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, seq_len, kernel_size)
        # self.attn_1 = SpatialAttention_1(dim, seq_len, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
        self.norm3 = nn.BatchNorm2d(dim)
        # self.cost = CoSTb(seq_len,seq_len,seq_len)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        # self.layer_scale_3 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'layer_scale_3', 'layer_scale_4'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        # x = x + self.drop_path(
        #     self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * self.attn_1(self.norm3(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        # print(x.shape)
        return x


class ConvMixerSubBlock(nn.Module):
    """A block of ConvMixer."""

    def __init__(self, dim, kernel_size=9, activation=nn.GELU):
        super().__init__()
        # spatial mixing
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        # channel mixing
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        x = x + self.norm_1(self.act_1(self.conv_dw(x)))
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, ls_init_value=1e-6, conv_mlp=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma'}

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) * self.mlp(self.norm(self.conv_dw(x))))
        return x


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4., drop_path=0.1, init_value=1e-6):
        super().__init__(dim, mlp_ratio=mlp_ratio, drop_path=drop_path, init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

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


class MLPMixerSubBlock(MixerBlock):
    """A block of MLP-Mixer."""

    def __init__(self, dim, input_resolution=None, mlp_ratio=4., drop=0., drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(dim, seq_len=seq_len,
                         mlp_ratio=(0.5, mlp_ratio), drop_path=drop_path, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, mlp_ratio=4., drop_rate=0., drop_path_rate=0., init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        # spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation=attn_dw_dilation, attn_channel_split=attn_channel_split)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path,
                         drop=drop, init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, drop=drop, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


def UniformerSubBlock(embed_dims, mlp_ratio=4., drop=0., drop_path=0.,
                      init_value=1e-6, block_type='Conv'):
    """Build a block of Uniformer."""

    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        return SABlock(dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                       drop=drop, drop_path=drop_path, init_value=init_value)


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                         init_value=init_value, act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class ViTSubBlock(ViTBlock):
    """A block of Vision Transformer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim=dim, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                         drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    

class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    

class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u
    
class TemporalAttentionModule_a(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool=nn.AdaptiveMaxPool2d(1)
        # self.avgpool=nn.AdaptiveAvgPool2d(1)
        # self.se=nn.Sequential(
        #     nn.Conv2d(dim,dim//reduction,1,bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(dim//reduction,dim,1,bias=False)
        # )
        # self.sigmoid=nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.GELU(),
            # nn.Linear(dim // self.reduction, dim // self.reduction, bias=False), # reduction
            # nn.GELU(),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn_1 = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn_1) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(f_x).view(b, c)+self.avg_pool(attn).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        # print(se_atten.shape,f_x.shape,u.shape)
        return se_atten * f_x * u



class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                 drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        
        self.attn = TemporalAttention(dim, kernel_size)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 5, 1, 2, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class MixMlp_3D(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        self.T = in_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
                # kaiming_init(m)
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b,self.T,c//self.T,h,w)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.reshape(b,c,h,w)
        return x





# ===================================================TripletAttention=======================================================

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out
    

    
# ===================================================Dan=======================================================
# 这个loss看起来比较好 但是R2不高
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y

class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class DAModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=d_model,kernel_size=kernel_size,H=H,W=W)
        self.channel_attention_module=ChannelAttentionModule(d_model=d_model,kernel_size=kernel_size,H=H,W=W)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out
    

# ===================================================CoordAtt=======================================================
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    

# ===========================================================TADA===================================================================
class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0] // 2, 0, 0],
        )
        self.bn = nn.BatchNorm3d(int(c_in // ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1] // 2, 0, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.

    def forward(self, x):
        g = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x


class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1
        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) * self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D,
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            elif len(x.shape) == 3:
                x = self.weight[:, None] * x + self.bias[:, None]
            return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
# class MVF(nn.Module):
#     """MVF Module"""
#     def __init__(self, in_channels, seq_len, alpha=0.5, use_hs=True, share=False):
#         super(MVF, self).__init__()
#         num_shift_channel = int(in_channels * alpha)
#         self.num_shift_channel = num_shift_channel
#         self.share = share
#         if self.num_shift_channel != 0:
#             self.split_sizes = [num_shift_channel, in_channels - num_shift_channel]

#             self.shift_conv = nn.Conv3d(
#                 num_shift_channel, num_shift_channel, [3, 1, 1], stride=1,
#                 padding=[1, 0, 0], groups=num_shift_channel, bias=False)

#             self.bn = nn.BatchNorm3d(num_shift_channel)
#             self.use_hs = use_hs
#             self.activation = HardSwish() if use_hs else nn.ReLU(inplace=True)

#             if not self.share:
#                 self.h_conv = nn.Conv3d(
#                     num_shift_channel, num_shift_channel, [1, 3, 1], stride=1,
#                     padding=[0, 1, 0], groups=num_shift_channel, bias=False)
#                 self.w_conv = nn.Conv3d(
#                     num_shift_channel, num_shift_channel, [1, 1, 3], stride=1,
#                     padding=[0, 0, 1], groups=num_shift_channel, bias=False)                 
#             self._initialize_weights()
#         self.conv_1 = nn.Conv3d(seq_len, seq_len, [1, 1, 1], stride=1,padding=[0, 0, 0])
#         self.conv_2 = nn.Conv3d(seq_len, seq_len, [1, 3, 3], stride=1,padding=[0, 1, 1])
#         self.conv_3 = nn.Conv3d(seq_len, seq_len, [1, 1, 1], stride=1,padding=[0, 0, 0])
#         self.T = seq_len
#         # print('=> Using Multi-view Fusion...')

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 # kaiming_init(m)
#                 n = m.kernel_size[0] * m.kernel_size[1] * \
#                     m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         """forward"""
#         b, c, h, w = x.size()
#         x = x.view(b, self.T, c//self.T, h, w).transpose(1, 2)  # n, c, t, h, w
#         x = list(x.split(self.split_sizes, dim=1))

#         # get H & W
#         if self.share:
#             tmp_h = self.shift_conv(x[0].transpose(2, 3)).transpose(2, 3)
#             tmp_w = self.shift_conv(x[0].permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
#         else:
#             tmp_h = self.h_conv(x[0])
#             tmp_w = self.w_conv(x[0])
#         x[0] = self.shift_conv(x[0]) + tmp_h + tmp_w

#         if self.use_hs:
#             # add bn and activation
#             x[0] = self.bn(x[0])
#             x[0] = self.activation(x[0])
#         x = torch.cat(x, dim=1)  # n, c, t, h, w
#         x = x.transpose(1, 2)
#         x = self.conv_1(x)
#         x = self.conv_2(x)
#         x = self.conv_3(x)
#         x = x.contiguous().view(b, c, h, w)
#         return x
    
# class HardSigmoid(nn.Module):
#     """h_sigmoid"""
#     def __init__(self, inplace=True):
#         super(HardSigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, x):
#         """forward"""
#         return self.relu(x + 3) / 6
# class HardSwish(nn.Module):
#     """h_swish"""
#     def __init__(self, inplace=True):
#         super(HardSwish, self).__init__()
#         self.sigmoid = HardSigmoid(inplace=inplace)

#     def forward(self, x):
#         """forward"""
#         return x * self.sigmoid(x)

def conv3v(input, weight, stride, padding, dilation, groups):
    weight = weight.squeeze(2)  # (Cout, Cin, 1, K, K) -> (Cout, Cin, K, K)
    padding_hw = (0, padding, padding)
    padding_tw = (padding, 0, padding)
    padding_th = (padding, padding, 0)
    hw = F.conv3d(input, weight.unsqueeze(2), None, stride, padding_hw,
                  dilation, groups)
    tw = F.conv3d(input, weight.unsqueeze(3), None, stride, padding_tw,
                  dilation, groups)
    th = F.conv3d(input, weight.unsqueeze(4), None, stride, padding_th,
                  dilation, groups)
    return hw, tw, th


class CoSTb(nn.Module):
    """CoST(b) module.

    https://arxiv.org/abs/1903.01197.

    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool): Same as nn.Conv3d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 seq_len,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _triple(stride)
        self.padding = padding
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = 'zeros'
        self.output_padding = (0, 0, 0)
        self.transposed = False

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, kernel_size,
                        kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc1 = nn.Linear(out_channels, out_channels, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.T = seq_len
        self.init_weights()

    def init_weights(self):
        kaiming_init(self)
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2, std=0.01)

    def forward(self, input):
        b,c,h,w = input.shape
        input = input.reshape(b,self.T,h,w,c//self.T)
        u = input.clone()
        hw, tw, th = conv3v(input, self.weight, self.stride, self.padding,
                            self.dilation, self.groups)
        # print(hw.shape, tw.shape, th.shape)

        pool_hw = self.max_pool(hw).view(-1, self.out_channels)  # (N, C)
        pool_tw = self.max_pool(tw).view(-1, self.out_channels)  # (N, C)
        pool_th = self.max_pool(th).view(-1, self.out_channels)  # (N, C)

        x = torch.concat((pool_hw, pool_tw, pool_th), dim=0)  # (3N, C)
        x = self.fc1(x)  # (3N, C)
        x = x.view(3, -1).permute((1, 0))  # (N*C, 3)
        x = self.fc2(x)  # (N*C, 3)
        alpha = x.permute((1, 0)).view(  # noqa
            3, -1, self.out_channels, 1, 1, 1)  # (3, N, C, 1, 1, 1)
        alpha = torch.softmax(alpha, dim=0)
        # print(hw.shape,tw.shape,th.shape)
        # print(alpha.shape)

        output = hw * alpha[0] + tw * alpha[1] + th * alpha[2]

        if self.bias is not None:
            output += self.bias.view(*self.bias.shape, 1, 1, 1)
        output = u*output
        output = output.reshape(b,c,h,w)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0, ) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
    
# ===================================================SequentialPolarizedSelfAttention_1=======================================================
class SequentialPolarizedSelfAttention_1(nn.Module):
    def __init__(self, T=10, channel=512):
        super().__init__()
        self.time_wv = nn.Conv2d(T, T//2, kernel_size=(1, 1))
        self.time_wq = nn.Conv2d(T, 1, kernel_size=(1, 1))
        self.time_wz = nn.Conv2d(T//2, T, kernel_size=(1, 1))
        self.agp_time = nn.AdaptiveAvgPool2d((1, 1))
        self.ln_t = nn.LayerNorm(T)
        self.softmax_time = nn.Softmax(1)
        # self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        # self.softmax_channel = nn.Softmax(1)
        # self.softmax_spatial = nn.Softmax(-1)
        # self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        # self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        # self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.T = T

        self.conv0h = nn.Conv2d(channel, channel, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=channel)
        self.conv0v = nn.Conv2d(channel, channel, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=channel)
        self.conv_spatial_h = nn.Conv2d(channel, channel, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=channel, dilation=3)
        self.conv_spatial_v = nn.Conv2d(channel, channel, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=channel, dilation=3)
        self.conv1 = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        b,tc,h,w = x.size()
        # print(b,tc,h,w)
        c = tc//self.T
        t = self.T
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # print(self.T)
        # print(b,c)
        # print(b*c,t,h,w)
        # # b, t, c, h, w = x.size()
        x = x.reshape(b*c,t,h,w)
        # Time-only Self-Attention
        time_wv = self.time_wv(x)  # bs,t,h,w
        time_wq = self.time_wq(x)  # bs,t,h,w
        time_wv = time_wv.reshape(b*c, t//2, -1)  # c,bs,h*w
        time_wq = time_wq.reshape(b*c, -1, 1)  # bs,1,t
        time_wq = self.softmax_time(time_wq)
        time_wz = torch.matmul(time_wv,time_wq).unsqueeze(-1)   # c,1,h*w
        time_weight = self.sigmoid(self.ln_t(self.time_wz(time_wz).reshape(b*c, t, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                        1).reshape(
            b*c, t, 1, 1)  # bs,c,1,1 # c,1,h,w
        time_out = time_weight * x
        time_out = time_out.view(b*t,c,h,w)

        u = time_out.clone()
        attn = self.conv0h(time_out)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        out = u*attn
        out = out.reshape(b,tc,h,w)
        return out

# ===================================================SequentialPolarizedSelfAttention_2=======================================================
class SequentialPolarizedSelfAttention_2(nn.Module):
    def __init__(self, T=10, channel=512):
        super().__init__()
        self.time_wv = nn.Conv2d(T, T//2, kernel_size=(1, 1))
        self.time_wq = nn.Conv2d(T, 1, kernel_size=(1, 1))
        self.time_wz = nn.Conv2d(T // 2, T, kernel_size=(1, 1))
        self.agp_time = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax_time = nn.Softmax(1)
        self.ln_t = nn.LayerNorm(T)
        # self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        # self.softmax_channel = nn.Softmax(1)
        # self.softmax_spatial = nn.Softmax(-1)
        # self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        # self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        # self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        # self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.conv0h = nn.Conv2d(channel, channel, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=channel)
        self.conv0v = nn.Conv2d(channel, channel, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=channel)
        self.conv_spatial_h = nn.Conv2d(channel, channel, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=channel, dilation=3)
        self.conv_spatial_v = nn.Conv2d(channel, channel, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=channel, dilation=3)
        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.T = T

    def forward(self, x):
        b,tc,h,w = x.size()
        c = tc//self.T
        t = self.T
        x = x.reshape(b*t,c,h,w)
        # Spatial-only Self-Attention
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        spatial_out = u*attn
        spatial_out = spatial_out.reshape(b*c,t,h,w)

        # Time-only Self-Attention
        time_wv = self.time_wv(spatial_out)  # bs,t,h,w
        time_wq = self.time_wq(spatial_out)  # bs,t,h,w
        time_wv = time_wv.reshape(b*c, t//2, -1)  # c,bs,h*w
        time_wq = time_wq.reshape(b*c, -1, 1)  # bs,1,t
        time_wq = self.softmax_time(time_wq)
        time_wz = torch.matmul(time_wv,time_wq).unsqueeze(-1)   # c,1,h*w
        time_weight = self.sigmoid(self.ln_t(self.time_wz(time_wz).reshape(b*c, t, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                        1).reshape(
            b*c, t, 1, 1)  # bs,c,1,1 # c,1,h,w
        time_out = time_weight * spatial_out
        time_out = time_out.view(b,tc,h,w)

        return time_out

        # # Channel-only Self-Attention
        # channel_wv = self.ch_wv(time_out)  # bs,c//2,h,w
        # channel_wq = self.ch_wq(time_out)  # bs,1,h,w
        # channel_wv = channel_wv.reshape(b*t, c // 2, -1)  # bs,c//2,h*w
        # channel_wq = channel_wq.reshape(b*t, -1, 1)  # bs,h*w,1
        # channel_wq = self.softmax_channel(channel_wq)
        # channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        # channel_weight = self.sigmoid(
        #     self.ln(self.ch_wz(channel_wz).reshape(b*t, c, 1).permute(0, 2, 1))).permute(0, 2,
        #                                                                                 1).reshape(
        #     b*t, c, 1, 1)  # bs,c,1,1
        # channel_out = channel_weight * time_out

        # # Spatial-only Self-Attention
        # spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        # spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        # spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        # spatial_wv = spatial_wv.reshape(b*t, c // 2, -1)  # bs,c//2,h*w
        # spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b*t, 1, c // 2)  # bs,1,c//2
        # spatial_wq = self.softmax_spatial(spatial_wq)
        # spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        # spatial_weight = self.sigmoid(
        #     spatial_wz.reshape(b*t, 1, h, w))  # bs,1,h,w
        # spatial_out = spatial_weight * channel_out
        # spatial_out = spatial_out.reshape(b,t,c,h,w)
        # return spatial_out

# ===================================================ParallelPolarizedSelfAttention_1=======================================================
class ParallelPolarizedSelfAttention_1(nn.Module):

    def __init__(self, T=10,channel=512):
        super().__init__()
        self.time_wv=nn.Conv2d(T,T//2,kernel_size=(1,1))
        self.time_wq=nn.Conv2d(T,1,kernel_size=(1,1))
        self.softmax_time=nn.Softmax(1)
        self.time_wz=nn.Conv2d(T//2,T,kernel_size=(1,1))
        self.ln_time=nn.LayerNorm(T)

        # self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        # self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        # self.softmax_channel=nn.Softmax(1)
        # self.softmax_spatial=nn.Softmax(-1)
        # self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        # self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        # self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        # self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        # self.agp=nn.AdaptiveAvgPool2d((1,1))

        self.conv0h = nn.Conv2d(channel, channel, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=channel)
        self.conv0v = nn.Conv2d(channel, channel, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=channel)
        self.conv_spatial_h = nn.Conv2d(channel, channel, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=channel, dilation=3)
        self.conv_spatial_v = nn.Conv2d(channel, channel, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=channel, dilation=3)
        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.T = T

    def forward(self, x):
        # b, c, h, w = x.size()
        b,tc,h,w = x.size()
        c = tc//self.T
        t = self.T
        x = x.reshape(b*c,t,h,w)
        #Time-only Self-Attention
        time_wv=self.time_wv(x) #bs,t//2,h,w
        time_wq=self.time_wq(x) #bs,1,h,w
        time_wv=time_wv.reshape(b*c,t//2,-1) #bs,t//2,h*w
        time_wq=time_wq.reshape(b*c,-1,1) #bs,h*w,1
        time_wq=self.softmax_time(time_wq)
        time_wz=torch.matmul(time_wv,time_wq).unsqueeze(-1) #bs,c//2,1,1
        time_weight=self.sigmoid(self.ln_time(self.time_wz(time_wz).reshape(b*c,t,1).permute(0,2,1))).permute(0,2,1).reshape(b*c,t,1,1) #bs,c,1,1
        time_out=time_weight*x
        # print(time_out.shape)
        time_out = time_out.reshape(b, tc, h, w)

        x = x.reshape(b*t,c,h,w)
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        out = u*attn
        spatial_out = out.reshape(b,tc,h,w)
        return spatial_out+time_out
        # x = x.reshape(b*t,c,h,w)
        # #Channel-only Self-Attention
        # channel_wv=self.ch_wv(x) #bs,c//2,h,w
        # channel_wq=self.ch_wq(x) #bs,1,h,w
        # channel_wv=channel_wv.reshape(b*t,c//2,-1) #bs,c//2,h*w
        # channel_wq=channel_wq.reshape(b*t,-1,1) #bs,h*w,1
        # channel_wq=self.softmax_channel(channel_wq)
        # channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        # channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b*t,c,1).permute(0,2,1))).permute(0,2,1).reshape(b*t,c,1,1) #bs,c,1,1
        # channel_out=channel_weight*x
        # channel_out = channel_out.reshape(b, t, c, h, w)

        # #Spatial-only Self-Attention
        # spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        # spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        # spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        # spatial_wv=spatial_wv.reshape(b*t,c//2,-1) #bs,c//2,h*w
        # spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b*t,1,c//2) #bs,1,c//2
        # spatial_wq=self.softmax_spatial(spatial_wq)
        # spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        # spatial_weight=self.sigmoid(spatial_wz.reshape(b*t,1,h,w)) #bs,1,h,w
        # spatial_out=spatial_weight*x
        # spatial_out = spatial_out.reshape(b, t, c, h, w)
        # out=spatial_out+channel_out+time_out
        # return out