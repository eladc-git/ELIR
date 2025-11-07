# =============================================================================
# Source: https://github.com/madebyollin/taesd/blob/main/taesd.py
# License: MIT License
#
# Attribution:
# This file was sourced from the repository "taesd",
# available at https://github.com/cszn/BSRGAN. Licensed under the MIT License: https://github.com/madebyollin/taesd/blob/main/LICENSE
# =============================================================================

#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, latent_channels):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
                        conv(3, 64), Block(64, 64),
                        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
                        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
                        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
                        conv(64, latent_channels))
    def forward(self, x):
        x = self.layers(x)
        return x

    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                sd_enc = state_dict["state_dict_enc"]
                self.load_state_dict(sd_enc)
            else:
                self.load_state_dict(state_dict)


class Decoder(nn.Module):
    def __init__(self, latent_channels, up_mode="nearest"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(conv(latent_channels, 64), nn.ReLU(),
                        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
                        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
                        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2, mode=up_mode), conv(64, 64, bias=False),
                        Block(64, 64), conv(64, 3))
    def forward(self, x):
        x = torch.tanh(x / 3) * 3
        return self.layers(x)

    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                sd_dec = state_dict["state_dict_dec"]
                self.load_state_dict(sd_dec)
            else:
                self.load_state_dict(state_dict)

class TAESD(nn.Module):

    def __init__(self, pretrained=True, latent_channels=16, up_mode='nearest'):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels, up_mode)
        if pretrained:
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            self.load_state_dict(pretrained.state_dict())

    def load_weights(self, path):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            if path.endswith(".ckpt"):
                sd_enc = state_dict["state_dict_enc"]
                self.encoder.load_state_dict(sd_enc)
                sd_dec = state_dict["state_dict_dec"]
                self.decoder.load_state_dict(sd_dec)
            else:
                self.load_state_dict(state_dict)

    def forward(self, x):
        '''
        Input in range [0,1]
        '''
        self.to(x.device)
        return self.decoder(self.encoder(x))