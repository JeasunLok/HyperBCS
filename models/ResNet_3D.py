import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from thop import profile
    
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
        # self.conv2 = ACmix(width, width, k_att, head, k_conv, stride=stride, dilation=dilation)
        self.conv2 = nn.Conv3d(width, width, kernel_size=(3, 3, 3), padding=(1, 1, 1))
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


class ResNet_3D(nn.Module):
    def __init__(self, block=Bottleneck, layers=[1], input_channels=32, k_att=7, head=4, k_conv=3, num_classes=11,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet_3D, self).__init__()

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

        self.bn1_3D = nn.BatchNorm3d(self.inplanes)

        self.relu_3D = nn.ReLU(inplace=True)

        self.maxpool_3D = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))


        self.layer1 = self.make_layer(block, 64, layers[0], k_att, head, k_conv)

        # self.layer2 = self.make_layer(block, 128, layers[1], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], k_att, head, k_conv, stride=2,
        #                                dilate=replace_stride_with_dilation[2])

        self.avgpool_3D = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        x = self.conv1_3D(x)
        x = self.bn1_3D(x)
        x = self.relu_3D(x)
        x = self.maxpool_3D(x)

        x = self.layer1(x)
        # x = self.layer2(x)

        x = self.avgpool_3D(x)
      
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    model = ResNet_3D().cuda()
    input = torch.randn([2,1,32,8,8]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(model(input).shape)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}M".format(flops / 1e6))
    print("params:{:.3f}M".format(params / 1e6))
    # --------------------------------------------------#
    #   用来测试网络能否跑通，同时可查看FLOPs和params
    # --------------------------------------------------#
    summary(model, input_size=(1,32,8,8), batch_size=-1)