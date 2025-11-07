# =============================================================================
# Source: https://github.com/cszn/BSRGAN/blob/main/utils/utils_blindsr.py
# License: MIT License
#
# Attribution:
# This file was sourced from the repository "BSRGAN",
# available at https://github.com/cszn/BSRGAN. Licensed under the Apache License: https://github.com/cszn/BSRGAN/blob/main/LICENSE
# =============================================================================

import torch
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import random
import torch.nn.functional as F



def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag


realesrgan_deg_cfg = {'resize_prob': [0.2, 0.7, 0.1], 'resize_range': [0.15, 1.5], 'gaussian_noise_prob': 0.5,
                        'noise_range': [1, 30], 'poisson_scale_range': [0.05, 3.0], 'gray_noise_prob': 0.4,
                        'jpeg_range': [30, 95], 'second_order_prob': 0.0, 'second_blur_prob': 0.8,
                        'resize_prob2': [0.3, 0.4, 0.3], 'resize_range2': [0.3, 1.2], 'gaussian_noise_prob2': 0.5,
                        'noise_range2': [1, 25], 'poisson_scale_range2': [0.05, 2.5], 'gray_noise_prob2': 0.4, 'jpeg_range2': [30, 95]}
jpeger = DiffJPEG(differentiable=False)

def degradation_realesrgan(data, patch_size=256, sf=4, resize_back=True):
    im_gt = data['gt'].unsqueeze(0)
    kernel1 = data['kernel1']
    kernel2 = data['kernel2']
    sinc_kernel = data['sinc_kernel']

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
        ['up', 'down', 'keep'],
        realesrgan_deg_cfg['resize_prob'],
    )[0]
    if updown_type == 'up':
        scale = random.uniform(1, realesrgan_deg_cfg['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(realesrgan_deg_cfg['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = realesrgan_deg_cfg['gray_noise_prob']
    if random.random() < realesrgan_deg_cfg['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=realesrgan_deg_cfg['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=realesrgan_deg_cfg['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*realesrgan_deg_cfg['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    if random.random() < realesrgan_deg_cfg['second_order_prob']:
        # blur
        if random.random() < realesrgan_deg_cfg['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            realesrgan_deg_cfg['resize_prob2'],
        )[0]
        if updown_type == 'up':
            scale = random.uniform(1, realesrgan_deg_cfg['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(realesrgan_deg_cfg['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
            mode=mode,
        )
        # add noise
        gray_noise_prob = realesrgan_deg_cfg['gray_noise_prob2']
        if random.random() < realesrgan_deg_cfg['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=realesrgan_deg_cfg['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=realesrgan_deg_cfg['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(ori_h // sf, ori_w // sf),
            mode=mode,
        )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*realesrgan_deg_cfg['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*realesrgan_deg_cfg['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(ori_h // sf, ori_w // sf),
            mode=mode,
        )
        out = filter2D(out, sinc_kernel)

    # resize back
    if resize_back:
        out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')

    # clamp and round
    im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

    # random crop
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, patch_size, 1 if resize_back else sf)
    lq, hq, flag_nan = replace_nan_in_batch(im_lq, im_gt)

    return lq.squeeze(0), hq.squeeze(0)