

import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import functools
import torch.nn.functional as F

import torch
from torch.nn import  Softmax


from .MV2Block_module import MV2Block
from .MobileViTBlock_module import MobileViTBlock
from .grid_attention_layer import GridAttentionBlock2D




def INF(B,H,W):
    # return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x



class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output



class DilatedParallelConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DilatedParallelConvBlock, self).__init__()
        assert out_planes % 4 == 0
        inter_planes = out_planes // 4
        self.conv1x1_down = nn.Conv2d(in_planes, inter_planes, 1, padding=0, groups=1, bias=False)
        self.conv1 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=1, dilation=1, groups=inter_planes, bias=False)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=2, dilation=2, groups=inter_planes, bias=False)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=4, dilation=4, groups=inter_planes, bias=False)
        self.conv4 = nn.Conv2d(inter_planes, inter_planes, 3, stride=stride, padding=8, dilation=8, groups=inter_planes, bias=False)
        self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.conv1x1_fuse = nn.Conv2d(out_planes, out_planes, 1, padding=0, groups=4, bias=False)
        self.attention = nn.Conv2d(out_planes, 4, 1, padding=0, groups=4, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.PReLU(out_planes)

    def forward(self, input):
        output = self.conv1x1_down(input)
        d1 = self.conv1(output)
        # print('d1 shape = ', d1.shape)
        d2 = self.conv2(output)
        # print('d2 shape = ', d2.shape)
        d3 = self.conv3(output)
        # print('d3 shape = ', d3.shape)
        d4 = self.conv4(output)
        # print('d4 shape = ', d4.shape)
        p = self.pool(output)
        d1 = d1 + p
        d2 = d1 + d2
        d3 = d2 + d3
        d4 = d3 + d4
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)
        output = self.conv1x1_fuse(torch.cat([d1, d2, d3, d4], 1))
        output = self.act(self.bn(output))

        return output




def split(x):
    c = int(x.size()[1])
    c1 = round(c // 2)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


class MobileViTSeg(nn.Module):
    def __init__(self, in_chns=1,class_num=2,P1=2, P2 =2,P3=2,  aux=False):
        super(MobileViTSeg, self).__init__()
        self.D1 = int(P1 / 2)
        print('P1 = ', P1)
        print('D1 = ', self.D1)
        self.D2 = int(P2 / 2)
        print('P2 = ', P2)
        print('D2 = ', self.D2)
        self.D3 = int(P3 / 2)
        print('P3 = ', P3)
        print('D3 = ', self.D3)


        # dims = [144, 192, 240]
        # dims = [64, 80, 96]
        # dims = [16, 32, 64]
        dims = [8, 8, 8]
        # dims = [2, 4, 2]

        self.aux = aux
        self.in_chns = in_chns

        # MV2Block(3, 8, stride=1)
        self.long1 = MV2Block(self.in_chns,8,stride=1)
        self.down1 = MV2Block(self.in_chns,8,stride=1)
        self.long1_02 = MV2Block(8, 8, stride=2)
        self.down1_02 = MV2Block(8, 8, stride=2)

        self.oneXone = nn.Conv2d(8,1,1,stride=1,padding=0,bias=False)
        # self.level1 = DilatedParallelConvBlock(8,8)

        self.level1 = nn.ModuleList()
        self.level1_long = nn.ModuleList()

        # MobileViTBlock(dims[0], 2, 8, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[0] * 2))
        for i in range(0, P1):
            self.level1.append(MV2Block(8, 8, stride=1))
        for i in range(0, self.D1):
            # self.level1_long.append(MobileViTBlock(dims[0], 2, 8, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[0] * 2)))
            # self.level1_long.append(MobileViTBlock(dims[0], 1, 8, kernel_size=3, patch_size=(8, 8), mlp_dim=int(dims[0] * 2)))
            self.level1_long.append(MobileViTBlock(dims[0], 3, 8, kernel_size=3, patch_size=(8, 8), mlp_dim=int(dims[0] / 2)))


        self.CC_plus_8 = CrissCrossAttention(in_dim=8)
        self.cat1 = nn.Sequential(
                    nn.Conv2d(16, 16, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(16))

        self.long2 = MV2Block(8, 24, stride=2)
        self.down2 = MV2Block(8, 24, stride=2)

        self.level2 = nn.ModuleList()
        self.level2_long = nn.ModuleList()

        # MobileViTBlock(dims[0], 2, 24, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[0] * 2))
        for i in range(0, P2):
            self.level2.append(MV2Block(24, 24, stride=1))
        for i in range(0, self.D2):
            # self.level2_long.append(MobileViTBlock(dims[1], 1, 24, kernel_size=3, patch_size=(4, 4), mlp_dim=int(dims[1] * 2)))
            self.level2_long.append(MobileViTBlock(dims[1], 3, 24, kernel_size=3, patch_size=(4, 4), mlp_dim=int(dims[1] /6)))

        self.CC_plus_24 = CrissCrossAttention(in_dim=24)
        self.cat2 = nn.Sequential(
                    nn.Conv2d(48, 48, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(48))

        self.long3 = MV2Block(24, 32, stride=2)
        self.down3 = MV2Block(24, 32, stride=2)

        # MobileViTBlock(dims[0], 2, 32, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[0] * 2))
        # self.level3 = MobileViTBlock(dims[2], 2, 32, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[2] * 2))

        self.level3 = nn.ModuleList()
        self.level3_long = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(MV2Block(32, 32, stride=1))
        for i in range(0, self.D3):
            # self.level3_long.append(MobileViTBlock(dims[2], 1, 32, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[2] * 2)))
            self.level3_long.append(MobileViTBlock(dims[2], 3, 32, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[2] /4)))

        self.CC_plus_32 = CrissCrossAttention(in_dim=32)
        self.cat3 = nn.Sequential(
                    nn.Conv2d(64, 64, 1, stride=1, padding=0, groups=1, bias=False),
                    nn.BatchNorm2d(64))

        # self.long4 = MV2Block(32, 64, stride=2)
        # self.down4 = MV2Block(32, 64, stride=2)
        #
        # # self.level4 = DilatedParallelConvBlock(64, 64, stride = 1)
        # # self.level4 = MobileViTBlock(dims[2], 2, 64, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[2] * 2))
        #
        # self.level4 = nn.ModuleList()
        # self.level4_long = nn.ModuleList()
        # for i in range(0, P4):
        #     self.level4.append(MV2Block(64, 64, stride=1))
        # for i in range(0, self.D4):
        #     self.level4_long.append(MobileViTBlock(dims[2], 2, 64, kernel_size=3, patch_size=(2, 2), mlp_dim=int(dims[2] * 2)))


        # self.up4_conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        # self.up4_bn4 = nn.BatchNorm2d(64)
        # self.up4_act = nn.PReLU(64)
        #
        # self.CC_plus_64 = CrissCrossAttention(in_dim=64)


        self.up3_conv4 = DilatedParallelConvBlockD2(64, 32)
        self.up3_conv3 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(32)
        self.up3_act = nn.PReLU(32)

        self.up2_conv3 = DilatedParallelConvBlockD2(32, 24)
        self.up2_conv2 = nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(24)
        self.up2_act = nn.PReLU(24)

        self.up1_conv2 = DilatedParallelConvBlockD2(24, 8)
        self.up1_conv1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(8)
        self.up1_act = nn.PReLU(8)



        self.GridAttentionBlock2D_8 = GridAttentionBlock2D(in_channels=8, inter_channels=2, gating_channels=8)
        self.GridAttentionBlock2D_24 = GridAttentionBlock2D(in_channels=24, inter_channels=2, gating_channels=24)
        self.GridAttentionBlock2D_32 = GridAttentionBlock2D(in_channels=32, inter_channels=2, gating_channels=32)

        if self.aux:
            self.pred1234 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(1, class_num, 1, stride=1, padding=0))
            self.pred123 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(3, class_num, 1, stride=1, padding=0))
            self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(64, class_num, 1, stride=1, padding=0))
            self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(32, class_num, 1, stride=1, padding=0))
            self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(24, class_num, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(8, class_num, 1, stride=1, padding=0))

    def forward(self, input):
        long1 = self.long1(input)
        output1 = self.down1(input)

        edge01 = self.oneXone(long1)
        # print('edge01 shape = ', edge01.shape)
        edge02 = self.oneXone(output1)
        # print('edge02 shape = ', edge02.shape)

        long1 = self.long1_02(long1)
        output1 = self.down1_02(output1)

        # output1_add = self.CC_plus_8(output1, long1)
        output1_add = output1 + long1
        # print('output1_add type = ', type(output1_add))
        # print('output1_add shape = ', output1_add.shape)

        for i, layer in enumerate(self.level1):
            if i < self.D1:
                output1 = layer(output1_add) + output1
                long1 = self.level1_long[i](output1_add) + long1
                output1_add = output1 + long1
            else:
                output1 = layer(output1_add) + output1
                # output1 = self.CC_plus_8(layer(output1_add), output1)
                output1_add = output1 + long1

        output1_cat = self.cat1(torch.cat([long1, output1], 1))
        output1_l, output1_r = split(output1_cat)

        # print('output1 type = ', type(output1))
        # print('output1 shape = ', output1.shape)
        # print('long1 shape = ', long1.shape)

        long2 = self.long2(output1_l + long1)
        output2 = self.down2(output1_r + output1)


        # output2_add = self.CC_plus_24(output2, long2)
        output2_add = output2 + long2
        # print('output2_add type = ', type(output2_add))
        # print('output2_add shape = ', output2_add.shape)

        for i, layer in enumerate(self.level2):
            if i < self.D2:
                output2 = layer(output2_add) + output2
                long2 = self.level2_long[i](output2_add) +  long2
                output2_add = output2 + long2
            else:
                output2 = layer(output2_add) + output2
                # output2 = self.CC_plus_24(layer(output2_add), output2)
                output2_add = output2 + long2


        # output2_add = output2 + long2

        output2_cat = self.cat2(torch.cat([long2, output2], 1))
        output2_l, output2_r = split(output2_cat)

        # print('output2 shape = ', output2.shape)
        # print('long2 shape = ', long2.shape)

        long3 = self.long3(output2_l + long2)
        output3 = self.down3(output2_r + output2)


        output3_add = output3 + long3
        # output3_add = self.CC_plus_32(output3, long3)

        for i, layer in enumerate(self.level3):
            if i < self.D3:
                output3 = layer(output3_add) + output3
                long3 = self.level3_long[i](output3_add) + long3
                output3_add = output3 + long3
            else:
                output3 = layer(output3_add) + output3
                output3_add = output3 + long3

        # output3 = self.CC_plus_32(self.level3(output3_add), output3)
        # output3_add = output3 + long3
        # print('output3_add type = ', type(output3_add))
        # print('output3_add shape = ', output3_add.shape)


        # output3_cat = self.cat3(torch.cat([long3, output3], 1))
        # output3_l, output3_r = split(output3_cat)

        # print('output3 shape = ', output3.shape)
        # print('long3 shape = ', long3.shape)


        up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        # up3_conv3 = self.CAM_plus_32(up3_conv3)
        # up3_conv3 = self.CAM_plus_32(up3_conv3)

        # print('up3_conv3 shape = ', up3_conv3.shape)
        # up3 = self.up3_act(up3_conv4 + up3_conv3)
        # multiply by factors


        # print('up3_conv4 shape = ', up3_conv4.shape)
        # print('up3_conv3 shape = ', up3_conv3.shape)

        up3 = self.up3_act(up3_conv3)

        up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        up2_conv3 = self.up2_conv3(up3)
        up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        # up2_conv2 = self.CAM_plus_24(up2_conv2)
        # up2_conv2 = self.CAM_plus_24(up2_conv2)
        # print('up2_conv2 shape = ', up2_conv2.shape)
        # up2 = self.up2_act(up2_conv3 + up2_conv2)
        # multiply by factors


        # print('up2_conv3 shape = ', up2_conv3.shape)
        # print('up2_conv2 shape = ', up2_conv2.shape)
        # up2_tmp, _ = self.GridAttentionBlock2D_24(up2_conv3, up2_conv2)
        # up2 = self.up2_act(up2_tmp)

        up2 = self.up2_act(self.CC_plus_24(up2_conv3, up2_conv2))

        up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        up1_conv2 = self.up1_conv2(up2)
        up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        # up1_conv1 = self.CAM_plus_8(up1_conv1)
        # up1_conv1 = self.CAM_plus_8(up1_conv1)
        # print('up1_conv1 shape = ', up1_conv1.shape)
        # up1 = self.up1_act(up1_conv2 + up1_conv1)

        # multiply by factors

        # print('up1_conv2 shape = ', up1_conv2.shape)
        # print('up1_conv1 shape = ', up1_conv1.shape)
        # up1_tmp, _ = self.GridAttentionBlock2D_8(up1_conv2, up1_conv1)
        # up1 = self.up1_act(up1_tmp)

        up1 = self.up1_act(self.CC_plus_8(up1_conv2, up1_conv1))
        # print('up1 shape = ', up1.shape)
        # print('output1.size() = ', output1.size())
        # print('up2 shape = ', up2.shape)
        # print('up3 shape = ', up3.shape)
        # print('up4 shape = ', up4.shape)

        if self.aux:
            pred3 = F.interpolate(self.pred3(up3), input.size()[2:], mode='bilinear', align_corners=False)
            pred2 = F.interpolate(self.pred2(up2), input.size()[2:], mode='bilinear', align_corners=False)
        pred1 = F.interpolate(self.pred1(up1), input.size()[2:], mode='bilinear', align_corners=False)

        if self.aux:
            return (pred1, pred2, pred3,edge01,edge02 )
        else:
            return pred1





# up1 shape = torch.Size([1, 8, 176, 176])
# up2 shape = torch.Size([1, 24, 176, 176])
# up3 shape = torch.Size([1, 32, 88, 88])
# up4 shape = torch.Size([1, 64, 44, 44])


# summary(MQattSeg(),(3,352,352))
#
#
# imgs = torch.randn(1, 3, 352,352)
#
# new_model = MobileViTSeg()
#
# output = new_model(imgs)
# #
# pred1, pred2, pred3, pred4,edge01,edge02 =  output
#
# print('output type = ', type(output))
# print('output length = ', len(output))
# print('pred1 shape = ', pred1.shape)
# print('pred2 shape = ', pred2.shape)
# print('pred3 shape = ', pred3.shape)
# print('pred4 shape = ', pred4.shape)