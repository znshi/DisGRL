from functools import partial
from timm.models import xception
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log
encoder_params = { #编码器关键字
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True) #partial:部分 pretrained（预训练） 设置为 True，会自动下载模型所对应权重，并加载到模型中
    }
}
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        #self.conv =SeparableConv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0
        
class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.initialize()


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

    def initialize(self):
        weight_init(self)
        
        
class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(728, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class DFA(nn.Module):
    """ Enhance the feature diversity.
    """
    def __init__(self, x, y):
        super(DFA, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )           
        self.conv2d = nn.Conv2d(y*3, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p1, p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

    def initialize(self):
        #pass
        weight_init(self)
        
class IEA(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, in_c, 1)
        self.conv2 = nn.Conv2d(in_c, in_c, 1)
        self.conv3 = nn.Conv2d(in_c, in_c, 1)
        self.conv0 = nn.Conv2d(in_c, in_c, 5, padding=2)
        self.conv1 = nn.Conv2d(in_c, in_c,3,padding=1)
        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)   
        self.norm_layer = nn.GroupNorm(4, in_c)   

    def forward(self, x):
        B,N,H,W=x.shape
        idn = x
        
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x,(x.shape[2],x.shape[3]))
        x = x-idn
        x0 = self.conv1(x)
        x1 = self.conv0(F.relu(self.bn1(x0),inplace=True))
      
       
        b, c, h, w = x1.size()
        x = x1.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x)
        return x

class IEA1(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, in_c, 1)
        self.conv2 = nn.Conv2d(in_c, in_c, 1)
        self.conv3 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        
        
        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.conv0 = nn.Conv2d(in_c, in_c, 5, padding=2)
        self.conv1 = nn.Conv2d(in_c, in_c,3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_c)        
        self.norm_layer = nn.GroupNorm(4, in_c)   

    def forward(self, x):
        B,N,H,W=x.shape
        idn = x
        
        x = self.conv1(x)
        #print('----------------:',x.shape)
        x = F.adaptive_avg_pool2d(x,(x.shape[2],x.shape[3]))
        
        x = x-idn
        
        x0 = self.conv0(x)
        
        x1 = self.conv1(F.relu(self.bn1(x0),inplace=True))
        
        x1_= idn+x1
        
        return x1_       #return x 
       
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class DGA(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
        
        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2)//2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )
        
        self.cw = nn.Conv2d(in_c, out_c, 1)
        
    def forward(self, lf,hf):
        if lf.size() != hf.size():
            lf = F.interpolate(lf, hf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf,hf),1)
        x1, x2 = self.w1(x), self.w2(x)
        out = self.wo(x1 * x2) + self.cw(x)
        return out

class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        #if c.size() != att.size():
          #  att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x
"""        
class GuidedAttention(nn.Module):

    def __init__(self, depth=728, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.reduce1 = Conv1x1(3,728)
        
        self.efm = EFM(728)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_full = self.reduce1(residual_full)
        res = self.efm(embedding,residual_full)
        return res

"""

class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=728, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.efm = EFM(728)

    def forward(self, x, pred_x,  embedding):   #pred_x2,
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        
        #residual_full2 = torch.abs(x - pred_x2)
        #residual_x2 = F.interpolate(residual_full2, size=embedding.shape[-2:],
                                   #mode='bilinear', align_corners=True)
        #res_map2 = self.gated(residual_x2)
        
        #res = (res_map + res_map2)*self.h(embedding) + self.dropout(embedding)
        res = self.efm(embedding,res_map)
        return res
        
class Attention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, inc, outc):
        super(Attention, self).__init__()
        self.conv1 = Conv1x1(inc,outc)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = Conv1x1(outc,outc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,(x.shape[2],x.shape[3]))
        #print('------------------------',x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x    

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out       
          
class DisGRL(nn.Module):
    """ Discrepancy-Guided Reconstruction Learning for Image Forgery Detection """

    def __init__(self, num_classes, drop_rate=0.2):
        super(DisGRL, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]() 
        
        self.eam = EAM()
        
        self.dfa1 = DFA(64, 32)
        self.dfa2 = DFA(128, 64)
        self.dfa3 = DFA(256, 128)
        self.dfa4 = DFA(728, 256)
        
        self.SAM = SAM()
        
        self.iea1 = IEA(64)
        self.iea2 = IEA(128+64)
        self.iea3 = IEA(256+128+64)
        self.iea4 = IEA(728+256+128+64)
        
        self.dga4 = DGA(984,256)
        self.dga3 = DGA(256+128,128)
        self.dga2 = DGA(128+64,64)
        self.dga1 = DGA(64+32,3)
        
        self.reduce1 = Conv1x1(16, 3)
        self.reduce2 = Conv1x1(256, 728)
        self.reduce3 = Conv1x1(1112, 728)
        self.reduce4 = Conv1x1(64+128+256+728, 728)
        
        self.reduce7 = Conv1x1(64, 3)
        self.reduce5 = Conv1x1(64, 32)
        self.reduce6 = Conv1x1(32, 3)
      
  
        
        self.att = Attention(827,728)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        #self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)
        self.fc = nn.Sequential(nn.Linear(encoder_params[self.name]["features"], 1024),nn.Hardswish(),nn.Linear(1024,num_classes))

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        
       
    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x):
        # clear the loss inputs
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x   # x: bt,3,299,299   # noise_x: bt,3,299,299
   
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)  # bt,32,149,149
        
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out1 = self.encoder.act2(out)   # bt,64,147,147
        
        out2 = self.encoder.block1(out1)  # bt,128,74,74
 
        out3 = self.encoder.block2(out2)   # bt,256,37,37
 
        out4 = self.encoder.block3(out3)  # bt,728,19,19
  
        out5 = self.encoder.block4(out4)   # bt,728,19,19
        
        dfa1 = self.dfa1(out1)
        dfa2 = self.dfa2(out2)
        dfa3 = self.dfa3(out3)
        dfa4 = self.dfa4(out4)
        
        
        iea1 = self.iea1(out1)
        
        iea1 = F.interpolate(iea1, size=out2.shape[-2:], mode='bilinear', align_corners=True)
        iea1 =  torch.cat((iea1,out2),1)
        iea2 = self.iea2(iea1)
        iea2 = F.interpolate(iea2, size=out3.shape[-2:], mode='bilinear', align_corners=True)
        iea2 =  torch.cat((iea2,out3),1)
        iea3 = self.iea3(iea2)
        
        iea3 = F.interpolate(iea3, size=out4.shape[-2:], mode='bilinear', align_corners=True)
        
        iea3 =  torch.cat((iea3,out4),1)
        iea4 = self.iea4(iea3)
        iea = self.reduce4(iea4)
  
        
        d4 = self.dga4(out5,dfa4) 
        d3 = self.dga3(d4,dfa3) 
        d2 = self.dga2(d3,dfa2)  #bt,64,74,74
        d1 = self.dga1(d2,dfa1) 
        
        d2_ = self.reduce5(d2)
        #print('--------------------',d2_.shape,dfa1.shape)
        sam_feature = self.SAM(d2_, dfa1)   #bt,32,74,74
        #print('--------------------',sam_feature.shape)
        sam_feature = self.reduce6(sam_feature)
        recons_x = F.interpolate(sam_feature, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        recons_x2 = F.interpolate(self.reduce7(d2), size=x.shape[-2:], mode='bilinear', align_corners=True)  #d1
        
        self.loss_inputs['recons'].append(recons_x)
        self.loss_inputs['recons'].append(recons_x2)
        
        
        norm_embed, corr = self.norm_n_corr(out5)
        self.loss_inputs['contra'].append(corr)
        
        #norm_embed, corr = self.norm_n_corr(d4)
        #self.loss_inputs['contra'].append(corr)

        
        embedding = self.encoder.block5(out5+iea) # bt,728,19,19   #out5+iea
        #print('--------------------',embedding.shape)
        embedding = self.encoder.block6(embedding) # bt,728,19,19
        embedding = self.encoder.block7(embedding) # bt,728,19,19
        #embedding = self.encoder.block8(embedding)
        
        img_att = self.attention(x, recons_x, embedding)
        img_att2 = self.attention(x, recons_x2, embedding)

        embedding = self.encoder.block9(img_att+img_att2)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)  # bt,1024,10,10
        
        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)  # bt,2048,10,10
    
        embedding = self.global_pool(embedding).squeeze()  # bt,2048
        
        out = self.dropout(embedding)  # bt,2048
        
        return self.fc(out) , recons_x ,recons_x2, iea
