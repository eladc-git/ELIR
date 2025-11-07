import torch
import torch.nn.functional as F
import math



def pos_emb(t, t_dim, scale=1000):
    assert t_dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
    if t.ndim < 1:
        t = t.unsqueeze(0)
    elif t.ndim > 1:
        t = t.squeeze()
    device = t.device
    half_dim = t_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb.detach()


def fm_loss(model, x_hq, x_lq, fm_cfg):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)

    with torch.no_grad():
        X_hq = model.enc(x_hq)
        X_lq = model.enc(x_lq)
        X_mmse = model.mmse(X_lq)

    b = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse + sigma_s * eps
    t = torch.rand([b, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim))

    loss = F.mse_loss(u, v)
    return loss


def cfm_loss(model, x_hq, x_lq, fm_cfg):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)

    with torch.no_grad():
        X_hq = model.enc(x_hq)
        X_lq = model.enc(x_lq)
        X_mmse = model.mmse(X_lq)

    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    # Flow Loss
    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim))

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim))

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends

    loss = F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_)

    return loss


def l2_fm_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    bs = x_hq.shape[0]

    # L2 loss
    with torch.no_grad():
        X_hq = tmodel.encoder(x_hq)
    X_lq = model.enc(x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow loss
    eps = torch.randn_like(X_hq)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim))

    loss += F.mse_loss(u, v)
    return loss


def l2_fm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    beta = fm_cfg.get("beta", 0.001)

    # L2 loss
    with torch.no_grad():
        X_hq = tmodel.encoder(x_hq)
    X_lq = model.enc(x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow loss
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_hq)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim))

    loss += (1-beta)*F.mse_loss(u, v)

    # MSE Loss
    X1 = Xt + (1 - t) * v
    x_hq_hat = tmodel.decoder(X1)
    loss += beta*F.mse_loss(x_hq, x_hq_hat)

    return loss


def l2_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)

    # L2 loss
    with torch.no_grad():
        X_hq = tmodel.encoder(x_hq)
    X_lq = model.enc(x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    # Flow Loss
    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim))

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim))

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends

    loss += (F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_))

    return loss


def pixel_space_l2_cfm_loss(model, x_hq, x_lq, fm_cfg):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)

    # L2 loss
    x_mmse = model.mmse(x_lq)
    loss = F.mse_loss(x_hq, x_mmse)

    bs = x_hq.shape[0]
    eps = torch.randn_like(x_mmse)
    X_mmse_noisy = x_mmse.detach() + sigma_s * eps
    t = (1 - dt) * torch.rand([bs, 1, 1, 1], device=x_hq.device, dtype=x_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K + 1, device=x_hq.device, dtype=x_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * x_hq

    # Flow Loss
    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * x_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim))

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * x_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim))

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less * (Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends

    loss += (F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_))

    return loss


def l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)
    beta = fm_cfg.get("beta", 0.001)

    # L2 loss
    with torch.no_grad():
        X_hq = tmodel.encoder(x_hq)
    X_lq = model.enc(x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow Loss
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim))

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim))

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends
    loss += (1-beta)*(F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_))

    # MSE loss
    f1 = f0.detach()
    v1 = model.fmir(f1, pos_emb(seg_ends, t_dim))
    X1 = f1 + (1 - seg_ends) * v1
    x_hq_hat = tmodel.decoder(X1)
    loss += beta*F.mse_loss(x_hq, x_hq_hat)

    return loss


def get_loss(model, x_hq, x_lq, fm_cfg, tmodel=None):
    method = fm_cfg.get("method")
    if method == "fm_loss":
        loss = fm_loss(model, x_hq, x_lq, fm_cfg)
    elif method == "cfm_loss":
        loss = cfm_loss(model, x_hq, x_lq, fm_cfg)
    elif method == "pixel_space_l2_cfm_loss":
        loss = pixel_space_l2_cfm_loss(model, x_hq, x_lq, fm_cfg)
    elif method == "l2_fm_loss":
        loss = l2_fm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_fm_mse_loss":
        loss = l2_fm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_loss":
        loss = l2_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_mse_loss":
        loss = l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    else:
        assert False, "Error: Unknown training method!"
    return loss
