import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class Block2D(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, activation=True, overparametrization=False):
        super().__init__()
        if overparametrization:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, 4*c_out, kernel_size=kernel, padding=1),
                nn.Conv2d(4*c_out, c_out, kernel_size=1),
                nn.SiLU() if activation else nn.Identity()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=kernel, padding=1),
                nn.SiLU() if activation else nn.Identity()
            )

    def forward(self, x):
        return self.block(x)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, c_in=16, c_hid=32, overparametrization=False):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = Block2D(c_in, c_hid, 3, True, overparametrization)
        self.conv2 = Block2D(c_in + c_hid, c_hid, 3, True, overparametrization)
        self.conv3 = Block2D(c_in + 2 * c_hid, c_hid, 3, True, overparametrization)
        self.conv4 = Block2D(c_in + 3 * c_hid, c_hid, 3, True, overparametrization)
        self.conv5 = Block2D(c_in + 4 * c_hid, c_in, 3, False, overparametrization)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, c_inout=16, c_hid=64, overparametrization=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)
        self.RDB2 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)
        self.RDB3 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    ''' RRBNet: Latent Reconstuction Module'''
    def __init__(self, c_inout=16, c_hid=64, n_rrdb=1, overparametrization=False):
        super(RRDBNet, self).__init__()
        self.overparametrization = overparametrization
        layers = []
        for _ in range(n_rrdb):
            layers.append(RRDB(c_inout, c_hid, overparametrization))
        self.lrm = nn.Sequential(*layers)

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
                    conv1, conv2 = m.block[0], m.block[1]
                    collapsed_conv = self.collapse_conv(conv1, conv2)
                    m.block[0] = collapsed_conv
                    m.block[1] = nn.Identity()
            self.overparametrization = False

    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                state_dict = state_dict["state_dict_mmse"]
            self.load_state_dict(state_dict)

    def forward(self, x):
        return x + self.lrm(x)