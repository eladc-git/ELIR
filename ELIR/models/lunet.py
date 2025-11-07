import torch
import torch.nn as nn
import math
import torch.nn.functional as F





class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_emb_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        out = self.linear2(x)
        return out


class Upsample(nn.Module):
    def __init__(self,  in_channels, out_channels, use_convtr):
        super().__init__()
        self.use_convtr = use_convtr
        if self.use_convtr:
            self.convtr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.use_convtr:
            x = self.convtr(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.avgpool(self.conv(x))


class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=32, overparametrization=False):
        super().__init__()
        if overparametrization:
            self.block = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, 4*out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.Conv2d(4*out_channels, out_channels, kernel_size=1, stride=1, padding="same")
            )
        else:
            self.block = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            )

    def forward(self, x):
        return self.block(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, overparametrization=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = Block2D(in_channels, out_channels, overparametrization=overparametrization)
        self.block2 = Block2D(out_channels, out_channels, overparametrization=overparametrization)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        h = self.block1(x)
        h += self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        out = h + self.conv2d(x)
        return out


class LUnet(nn.Module):
    def __init__(self, ch_mult=[1,2,1,2], n_mid_blocks=3, in_channels=16, hid_channels=128,
                 out_channels=16, t_emb_dim=160, use_rescale_conv=True, overparametrization=False):
        super(LUnet, self).__init__()
        self.overparametrization = overparametrization
        self.t_emb_dim = t_emb_dim
        time_dim_out = 4*t_emb_dim
        self.time_mlp = TimestepEmbedding(in_channels=t_emb_dim, time_emb_dim=time_dim_out)
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.first_proj = nn.Conv2d(in_channels, hid_channels, kernel_size=1)

        # Down blocks
        chs = hid_channels
        for mult in ch_mult:
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization)
            if mult!=1:
                downsample = Downsample(chs, mult * chs, use_conv=use_rescale_conv)
                chs = mult * chs
            else:
                downsample = nn.Identity()
            self.down_blocks.append(nn.ModuleList([resnet, downsample]))

        # Mid blocks
        for i in range(n_mid_blocks):
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization)
            self.mid_blocks.append(resnet)

        # Up blocks
        for mult in ch_mult[::-1]:
            if mult!=1:
                upsample = Upsample(chs, chs//mult, use_convtr=use_rescale_conv)
                chs = chs // mult
            else:
                upsample = nn.Identity()
            resnet = ResnetBlock2D(2*chs, chs, time_dim_out, overparametrization=overparametrization)
            self.up_blocks.append(nn.ModuleList([upsample, resnet]))

        self.final_block = Block2D(chs, chs, overparametrization=overparametrization)
        self.final_proj = nn.Conv2d(chs, out_channels, kernel_size=1)

    def reset(self):
        for n, m in self.named_modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def collapse_conv(self, conv1, conv2):
        collapsed_conv = nn.Conv2d(conv1.in_channels,
                                   conv2.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding)
        kx, ky = conv1.weight.shape[2] + conv2.weight.shape[2] - 1, conv1.weight.shape[3] + conv2.weight.shape[3] - 1
        x_pad, y_pad = 2 * kx - 1, 2 * ky - 1
        in_tensor = torch.eye(conv1.weight.shape[1], device=conv1.weight.device)
        in_tensor = torch.unsqueeze(torch.unsqueeze(in_tensor, 2), 3)
        in_tensor = F.pad(in_tensor, (int(math.ceil((x_pad - 1) / 2)),
                                      int(math.floor((x_pad - 1) / 2)),
                                      int(math.ceil((y_pad - 1) / 2)),
                                      int(math.floor((y_pad - 1) / 2))))
        # Run first Conv2D
        conv1_out = F.conv2d(input=in_tensor, weight=conv1.weight, stride=conv1.stride, padding=(0, 0))
        # Run second Conv2D
        conv2_out = F.conv2d(input=conv1_out, weight=conv2.weight, stride=conv2.stride)
        # Extract collapsed kernel from output: the collapsed kernel is the output of the convolution after fixing the dimension
        collapsed_kernel = torch.permute(torch.flip(conv2_out, [3, 2]), dims=[1, 0, 2, 3])
        collapsed_bias = torch.matmul(torch.sum(conv2.weight, dim=(2, 3)), conv1.bias) + conv2.bias
        sd = {"weight": collapsed_kernel, "bias": collapsed_bias}
        collapsed_conv.load_state_dict(sd)
        return collapsed_conv

    def collapse(self):
        if self.overparametrization:
            for _, m in self.named_modules():
                if isinstance(m, Block2D):
                    conv1, conv2 = m.block[2], m.block[3]
                    collapsed_conv = self.collapse_conv(conv1, conv2)
                    m.block[2] = collapsed_conv
                    m.block[3] = nn.Identity()
            self.overparametrization = False

    def load_weights(self, model_path):
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            if model_path.endswith(".ckpt"):
                state_dict = state_dict["state_dict_fmir"]
            self.load_state_dict(state_dict)

    def forward(self, xt, t_emb):
        emb = self.time_mlp(t_emb)
        x = self.first_proj(xt)
        # Down blocks
        skip_connect = []
        for resnet, downsample in self.down_blocks:
            x = resnet(x, emb)
            skip_connect.append(x)
            x = downsample(x)
        # Mid blocks
        for resnet in self.mid_blocks:
            x = resnet(x, emb)
        # Up blocks
        for upsample, resnet in self.up_blocks:
            x = upsample(x)
            x = torch.concat([x,skip_connect.pop()], dim=1)
            x = resnet(x, emb)
        x = self.final_block(x)
        x = self.final_proj(x)
        return x
