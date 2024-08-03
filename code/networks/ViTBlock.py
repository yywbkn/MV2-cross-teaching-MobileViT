

import torch.nn as nn
import functools

import torch
from thop import profile
from thop import clever_format
from einops import rearrange

def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('#'*20, '\n[Statistics Information]\nFLOPs: {}\nParams: {}\n'.format(flops, params), '#'*20)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class ViTParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ViTParallelConvBlock, self).__init__()
        assert out_planes % 2 == 0
        inter_planes = out_planes // 2
        # print('inter_planes = ',inter_planes)
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 2, 1, padding=0, groups=2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)
        if out_planes <= 8:
            self.h = 16
        elif out_planes == 24:
            self.h = 4
        else:
            self.h = 2

        self.new_model = MobileViTBlock(dim = out_planes*2 ,depth =1,channel = inter_planes, kernel_size=3, patch_size=(self.h, self.h),mlp_dim = int(out_planes * 4))


    def forward(self, input):
        output = self.conv1x1_down(input)
        p = self.pool(output)

        d1 = self.conv1(output)
        # print('d1 shape = ', d1.shape)
        d1 = self.new_model(d1)
        # print('d1 shape = ', d1.shape)
        d2 = self.conv2(output)
        # print('d2 shape = ', d2.shape)
        d2 = self.new_model(d2)
        # print('d2 shape = ', d2.shape)
        d1 = d1 + p
        d2 = d1 + d2
        att = torch.sigmoid(self.attention(torch.cat([d1, d2], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2], 1))
        output = self.act(self.bn(output))

        return output









# trainsize = 352
# x = torch.randn(1, 3, trainsize, trainsize)
# model = ViTParallelConvBlock(3,8,stride=1)
# CalParams(model, x)

# imgs = torch.randn(1,  3, 352,352)
# new_model = ViTParallelConvBlock(3,8,stride=1)
# new_model = DilatedParallelConvBlock(3,8,stride=2)

# imgs = torch.randn(1, 8,352,352)
# new_model = ViTParallelConvBlock(8,24,stride=1)

# imgs = torch.randn(1, 8,176,176)
# new_model = ViTParallelConvBlock(8,24,stride=1)

# imgs = torch.randn(1, 24,88,88)
# new_model = ViTParallelConvBlock(24,32,stride=1)
#
# imgs = torch.randn(1, 32,44,44)
# new_model = ViTParallelConvBlock(32,32,stride=1)


# output = new_model(imgs)
#
# print('output type = ', type(output))
# print('output length = ', len(output))
# print('output shape = ', output.shape)
