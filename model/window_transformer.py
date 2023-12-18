import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from einops import rearrange, repeat

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_Enhance_DW(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) #设置网络中的全连接层
        self.act1 = nn.ReLU(inplace=True) #激活函数，克服梯度消失的问题，加快训练速度
        self.norm0 = nn.LayerNorm(hidden_features)#归一化操作

        self.depth_conv1 = nn.Conv2d(hidden_features,hidden_features,kernel_size=3,stride=1,groups=hidden_features,padding=1)
        self.norm1 = nn.LayerNorm(hidden_features)

        self.depth_conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, groups=hidden_features//4, padding=1)
        self.norm2 = nn.LayerNorm(hidden_features)

        self.norm3 = nn.LayerNorm(hidden_features)
        self.act2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act3 = nn.ReLU(inplace=True)
        self.norm4 = nn.LayerNorm(out_features)
        # self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.norm0(x)
        x = self.act1(x).permute(0, 2, 1).reshape(B, 2*C, H, W).contiguous()

        x1 = self.depth_conv1(x)
        x1 += x
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.norm1(x1).permute(0, 3, 1, 2).contiguous()

        x2 = self.depth_conv2(x1)
        x2 += x1
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.norm2(x2).permute(0, 3, 1, 2).contiguous()

        x2 += x
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.norm3(x2).permute(0, 3, 1, 2).contiguous()

        x = self.act2(x2).flatten(2).transpose(1,2)
        x = self.fc2(x)
        x = self.norm4(x)
        x = self.act3(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x

    # def flops(self, H, W):
    #     flops = 0
    #     # fc1
    #     flops += H*W*self.dim*self.hidden_dim
    #     # dwconv
    #     flops += H*W*self.hidden_dim*3*3
    #     # fc2
    #     flops += H*W*self.hidden_dim*self.dim
    #     print("LeFF:{%.2f}"%(flops/1e9))
    #     return flops

def window_partition(x, window_size):#Window Partition函数是用于对张量划分窗口，指定窗口大小。
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) #对x重构成6个维度
    # permute()将tensor的维度换位；contiguous()深拷贝，把换位后的tensor重新生成一份；view(-1，..),重构，其中第一个维度自动补齐
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)      #torch.Size([64, 7, 7, 96])  表示分为64个窗口，每个窗口的像素个数为49
    return windows


def window_reverse(windows, window_size, H, W): #Window Partition的逆过程
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  #torch.Size([1, 56, 56, 96])
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH   torch.Size([169, 3])

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  #Wh*Ww, Wh*Ww   #然后再最后一维上进行求和，展开成一个一维坐标，并注册为一个不参与网络学习的变量  torch.Size([49, 49])
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)  #torch.Size([169, 3])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # make torchscript happy (cannot use tensor as tuple)  torch.Size([64, 3, 49, 32])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))   #torch.Size([64, 3, 49, 49])

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]   #49*49 = 2401  torch.Size([2401, 3])
        relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH  shape
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  #torch.Size([64, 49, 96])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim,  num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True,  drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_mlp='tffn'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        # self.token_mlp = token_mlp

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)   #384
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = Mlp_Enhance_DW(in_features=dim,hidden_features=dim*2,out_features=dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,drop=drop) if token_mlp == 'ffn' else LeFF(dim, dim*4, act_layer=act_layer, drop=drop)

    def forward(self,x,H,W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size" #断言语句，如果不满足条件则直接触发异常，不必执行接下来的代码：

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  #torch.Size([1, 56, 56, 96])  重构成BHWC格式

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C   torch.Size([64, 49, 96])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        x = x.view(B, H * W, C)

        # FFN
        # x = x + self.mlp(self.norm2(x),H,W)
        # x = shortcut + self.mlp(self.norm2(x),H,W)
        x = shortcut + self.mlp(self.norm2(x))

        return x
