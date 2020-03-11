
import torch
import torch.nn as nn

from models import Resnet_Encoder




class FullVisionModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.contrastive_samples = self.opt.negative_samples
        self.inplanes = [64, 256, 512]
        self.channels = [64, 128, 256]
        self.num_blocks = [3, 4, 6]
        self.encode_num = range(3)

        self.encoder = nn.ModuleList([])
        for i in self.encode_num:
            self.encoder.append(Resnet_Encoder.ResNet(
                opt, Resnet_Encoder.Bottleneck, self.inplanes[i], self.channels[i], self.num_blocks[i], i
            ))

        print(self.encoder)

    def forward(self, x, n=3):
        model_input = x


        if self.opt.device.type != "cpu":
            cur_device = x.get_device()
        else:
            cur_device = self.opt.device

        loss = torch.ones(1, 3, device=cur_device)

        for idx, module in enumerate(self.encoder[:n+1]):
            h, z, cur_loss = module(model_input)
            model_input = z.detach()
            loss[:, idx] = cur_loss
        return loss, h
