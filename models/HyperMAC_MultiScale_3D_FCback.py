import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile

# transformer module positional embeddings fill with [-1,1]
def position_3D(C, H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1).repeat(C, 1, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W).repeat(C, 1, 1)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1).repeat(C, 1, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W).repeat(C, 1, 1)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    b, l, c, h, w = x.shape
    return x[:, :, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

# ACmix module
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1_3D = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1))
        self.conv2_3D = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1))
        self.conv3_3D = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1))
        self.conv_p_3D = nn.Conv3d(2, self.head_dim, kernel_size=(1, 1, 1))

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att_3D = torch.nn.ReflectionPad3d((self.padding_att,self.padding_att,self.padding_att,self.padding_att,0,0))

        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc_3D = nn.Conv3d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=(1, 1, 1), bias=False)
        self.dep_conv_3D = nn.Conv3d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        nn.init.kaiming_normal_(self.dep_conv_3D.weight, mode='fan_out', nonlinearity='relu')
        self.dep_conv_3D.bias = init_rate_0(self.dep_conv_3D.bias)

    def forward(self, x):
        q, k, v = self.conv1_3D(x), self.conv2_3D(x), self.conv3_3D(x)
        scaling = float(self.head_dim) ** -0.5
        b, l, c, h, w = q.shape

        h_out, w_out = h//self.stride, w//self.stride

        pe = self.conv_p_3D(position_3D(c, h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, c, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, c, h, w)
        v_att = v.view(b*self.head, self.head_dim, c, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_temp = self.pad_att_3D(k_att)
        unfold_k = torch.tensor([]).cuda()
        for i in range (unfold_temp.shape[2]):
            unfold_k_temp = self.unfold(unfold_temp[:,:,i,:,:]).unsqueeze(0)
            unfold_k = torch.concat([unfold_k, unfold_k_temp], 0)

        unfold_temp = self.pad_att_3D(pe)
        unfold_rpe = torch.tensor([]).cuda()
        for i in range (unfold_temp.shape[2]):
            unfold_rpe_temp = self.unfold(unfold_temp[:,:,i,:,:]).unsqueeze(0)
            unfold_rpe = torch.concat([unfold_rpe, unfold_rpe_temp], 0)

        c_out = unfold_temp.shape[2]//self.stride

        unfold_k = unfold_k.view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, c_out, h_out, w_out)
        unfold_rpe = unfold_rpe.view(1, self.head_dim, self.kernel_att*self.kernel_att, c_out, h_out, w_out)

        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) 
        att = self.softmax(att)

        unfold_temp = self.pad_att_3D(v_att)
        unfold_v = torch.tensor([]).cuda()
        for i in range (unfold_temp.shape[2]):
            unfold_v_temp = self.unfold(unfold_temp[:,:,i,:,:]).unsqueeze(0)
            unfold_v = torch.concat([unfold_v, unfold_v_temp], 0)

        out_att = unfold_v.view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, c_out, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, c_out, h_out, w_out)


        f_all = self.fc_3D(torch.cat([q.view(b, self.head, self.head_dim, c, h*w), k.view(b, self.head, self.head_dim, c, h*w), v.view(b, self.head, self.head_dim, c, h*w)], 1))

        f_conv = f_all.permute(0, 2, 1, 3, 4).reshape(x.shape[0], -1, x.shape[-3], x.shape[-2], x.shape[-1])
     
        out_conv = self.dep_conv_3D(f_conv)
        return self.rate1 * out_att + self.rate2 * out_conv
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, k_att, head, k_conv, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_3D = nn.Conv3d(inplanes, width, kernel_size=(1, 1, 1))
        norm_layer = nn.BatchNorm3d

        self.bn1 = norm_layer(width)
        self.conv2 = ACmix(width, width, k_att, head, k_conv, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(width)

        self.conv3_3D = nn.Conv3d(width, planes * self.expansion, kernel_size=(1, 1, 1))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_3D(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3_3D(out)
        out = self.bn3(out)
  
        if self.downsample is not None:
           identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HyperMAC_3D_MultiScale_FCback(nn.Module):
    def __init__(self, block=Bottleneck, layers=[1], input_channels=32, k_att=7, head=4, k_conv=3, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(HyperMAC_3D_MultiScale_FCback, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1_3D = nn.Conv3d(1, self.inplanes, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv1_3D_multiscale = nn.Conv3d(1, self.inplanes, kernel_size=(7, 7, 7), stride=(2, 1, 1), padding=(3, 3, 3), bias=False)

        self.bn1_3D = nn.BatchNorm3d(self.inplanes)

        self.relu_3D = nn.ReLU(inplace=True)

        self.maxpool_3D = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.maxpool_3D_multiscale = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.layer1 = self.make_layer(block, 64, layers[0], k_att, head, k_conv)

        # self.layer2 = self.make_layer(block, 128, layers[1], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[2])

        self.avgpool_3D = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.scale1_rate = torch.nn.Parameter(torch.Tensor(1))
        self.scale2_rate = torch.nn.Parameter(torch.Tensor(1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.scale1_rate)
        init_rate_half(self.scale2_rate)

    def make_layer(self, block, planes, blocks, rate, k, head, stride=1, dilate=False):
        norm_layer = nn.BatchNorm3d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, rate, k, head, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate, k, head, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x_multiscale = x[:, :, :, x.shape[-2]//4:(3*(x.shape[-2])//4), x.shape[-1]//4:(3*(x.shape[-1])//4)]

        x = self.conv1_3D(x)
        x_multiscale = self.conv1_3D(x_multiscale)

        x = self.bn1_3D(x)
        x_multiscale = self.bn1_3D(x_multiscale)

        x = self.relu_3D(x)
        x_multiscale = self.relu_3D(x_multiscale)

        x = self.maxpool_3D(x)
        x_multiscale = self.maxpool_3D_multiscale(x_multiscale)
 
        x = self.layer1(x)
        x_multiscale = self.layer1(x_multiscale)
        # x = self.layer2(x)

        x = self.avgpool_3D(x)
        x_multiscale = self.avgpool_3D(x_multiscale)

        x = torch.flatten(x, 1)
        x_multiscale = torch.flatten(x_multiscale, 1)

        x = self.fc(x)
        x_multiscale = self.fc(x_multiscale)

        output_x = self.scale1_rate * x + self.scale2_rate * x_multiscale

        return output_x


if __name__ == '__main__':
    model = HyperMAC_3D_MultiScale_FCback().cuda()
    input = torch.randn([2,1,32,8,8]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops / 1e9))
    print("params:{:.3f}M".format(params / 1e6))
    # --------------------------------------------------#
    #   用来测试网络能否跑通，同时可查看FLOPs和params
    # --------------------------------------------------#
    summary(model, input_size=(1,32,8,8), batch_size=-1)