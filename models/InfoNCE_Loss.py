import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE_Loss(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super().__init__()
        self.opt = opt
        self.negative_samples = self.opt.negative_samples
        self.k_predictions = self.opt.prediction_step

        self.W_k = nn.ModuleList([])
        for i in range(self.k_predictions):
            self.W_k.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, ztk, zt):
        batch_size = zt.shape[0]
        total_loss = 0

        if self.opt.device.type != "cpu":
            cur_device = zt.get_device()
        else:
            cur_device = self.opt.device

        # For each element in c, contrast with elements below
        for k in range(1, self.k_predictions + 1):
            ztwk = (
                self.W_k[k - 1]
                .forward(ztk[:, :, (k + 1):, :])  # BS, C , H , W
                .permute(2, 3, 0, 1)  # H, W, BS, C
                .contiguous()
            )  # H, W, BS, C

            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # H * W * BS, C
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # H * W * BS
                (ztwk_shuf.shape[0] * self.negative_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )

            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(ztwk_shuf, dim=0, index=rand_index, out=None)  # H * W * BS * n, C

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.negative_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # H, W, BS, C, n

            zt_compute = (zt[:, :, : -(k + 1), :].permute(2, 3, 0, 1).unsqueeze(-2))  # H, W, BS, 1, C

            log_fk_pos = torch.matmul(zt_compute, ztwk.unsqueeze(-1)).squeeze(-2)  # H, W, BS, 1
            log_fk_neg = torch.matmul(zt_compute, ztwk_shuf).squeeze(-2)  # H, W, BS, n

            log_fk = torch.cat((log_fk_pos, log_fk_neg), 3)  # H, W, BS, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # BS, 1+n, H, W

            log_fk_soft = torch.softmax(log_fk, dim=1)

            true_f = torch.zeros(
                (batch_size, log_fk_soft.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # BS, H, W

            contrast_loss = F.nll_loss(torch.log(log_fk_soft + 1e-11), true_f, ignore_index=-100, reduction='mean')
            total_loss += contrast_loss

        total_loss /= self.k_predictions

        return total_loss


