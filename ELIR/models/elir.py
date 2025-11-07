import math
import torch
import torch.nn as nn
from ELIR.models.load_model import get_model



def pos_emb(t, t_dim, scale=1000):
    assert t_dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
    t = torch.tensor([t])
    half_dim = t_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim,device=t.device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb


class Elir(nn.Module):
    def __init__(self, fm_cfg, fmir_cfg, mmse_cfg, enc_cfg, dec_cfg):
        super(Elir, self).__init__()
        self.fmir_cfg = fmir_cfg
        self.mmse_cfg = mmse_cfg
        self.enc_cfg = enc_cfg
        self.dec_cfg = dec_cfg
        self.K = fm_cfg.get("k_steps")
        self.latent_shape = fm_cfg.get("latent_shape")
        self.sigma_s = fm_cfg.get("sigma_s",0.1)
        self.dynamic_noise = fm_cfg.get("dynamic_noise",True)
        self.t_emb_dim = fmir_cfg.get("t_emb_dim",160)
        self.dt = 1/self.K
        self.fmir = get_model(fmir_cfg)
        self.mmse = get_model(mmse_cfg)
        self.enc = get_model(enc_cfg)
        self.dec = get_model(dec_cfg)
        self.noise = self.sigma_s * torch.randn((1, *self.latent_shape))

    def collapse(self):
        self.fmir.collapse()
        self.mmse.collapse()

    def load_weights(self, path):
        if path:
            state_dict = torch.load(path, map_location="cuda")
            if path.endswith(".ckpt"):
                sd_fmir = state_dict["state_dict_fmir"]
                self.fmir.load_state_dict(sd_fmir)
                sd_mmse = state_dict["state_dict_mmse"]
                self.mmse.load_state_dict(sd_mmse)
                sd_enc = state_dict["state_dict_enc"]
                self.enc.load_state_dict(sd_enc)
                sd_dec = state_dict["state_dict_dec"]
                self.dec.load_state_dict(sd_dec)
                self.collapse()
            else:
                self.load_state_dict(state_dict)

    def forward(self, x):
        self.to(x.device)
        z = self.enc(x)
        if self.dynamic_noise:
            noise = self.sigma_s*torch.randn_like(z)
        else:
            noise = self.noise
        z0 = self.mmse(z) + noise.to(x.device)
        dt = 0
        for k in range(self.K):
            z0 += self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
            dt += self.dt
        y = self.dec(z0)
        return y

    def inference(self, x):
        y = self(x)
        out = torch.clip(y, min=0, max=1)
        return out

    def trajectories_pixel(self, x):
        self.to(x.device)
        z = self.enc(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        trajs = [self.dec(z0.clone())]
        dt = 0
        for k in range(self.K):
            z0 += self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
            dt += self.dt
            trajs.append(self.dec(z0).clone())
        return trajs

    def trajectories(self, x):
        self.to(x.device)
        z = self.enc(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        trajs = [z0.clone()]
        dt = 0
        for k in range(self.K):
            z0 += self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
            dt += self.dt
            trajs.append(z0.clone())
        return trajs