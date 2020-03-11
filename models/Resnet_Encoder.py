import torch.nn as nn
import torch.nn.functional as F

from models import InfoNCE_Loss




class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1)
        if stride != 1 or inplanes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inplanes, expansion * planes, kernel_size=1, stride=stride
                )
            )
    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, opt, block, inplanes, channels, num_blocks, encode_num, input_dims=3):
        super(ResNet, self).__init__()
        self.expansion = 4
        self.opt = opt
        self.encode_num = encode_num
        self.model = nn.Sequential()

        if self.encode_num == 0:
            self.model.add_module("conv1", nn.Conv2d(input_dims, 64, kernel_size=5, stride=1, padding=2))
            self.model.add_module("conv {}".format(encode_num),
                                  self._make_layer(block, inplanes, channels, num_blocks, stride=1))
        else:
            self.model.add_module("conv {}".format(encode_num),
                                  self._make_layer(block, inplanes, channels, num_blocks, stride=2))

        self.loss = InfoNCE_Loss.InfoNCE_Loss(
            opt, in_channels=channels * self.expansion, out_channels=channels * self.expansion
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride, expansion=4):
        layers = []
        layers.append(block(inplanes, planes, stride, expansion))
        for i in range(1, blocks):
            layers.append(block(planes * expansion, planes, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.encode_num == 0:
            x = (x.unfold(2, 16, 8).unfold(3, 16, 8).permute(0, 2, 3, 1, 4, 5))
            x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5])

        z = self.model(x)

        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, 7, 7, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()

        loss = self.loss(out, out)

        return out, z, loss

