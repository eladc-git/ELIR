from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision.io import read_image
import os
from ELIR.utils import ImageSpliterTh
from ELIR.models.load_model import get_model
from tqdm import tqdm
import math
from utils import get_device
import glob
import warnings
warnings.filterwarnings("ignore")
device = get_device()

IMAGE_EXTENSION = ('jpg','png','jpeg')


def to_tensor(img_tensor):
    img_tensor = img_tensor[:3,...]
    img_tensor = img_tensor.unsqueeze(0) / 255.0
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    img_tensor = img_tensor.to(device)
    return img_tensor

def run_infer(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)


    # ----------------------------
    # Infer all images in folder
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    in_image_folder = eval_cfg.get("in_folder",[])
    img_size = eval_cfg.get("image_size",512)
    chop = eval_cfg.get("chop",None)
    out_image_folder = eval_cfg.get("out_folder","out")
    os.makedirs(out_image_folder, exist_ok=True) # run folder
    if chop:
        sf = chop.get("sf", 4)
        upscale = chop.get("upscale", 4)
        chop_size = chop.get("chop_size", 256)
        chop_stride = chop.get("chop_stride", 224)

    for img_path in tqdm(sorted(glob.glob(os.path.join(in_image_folder,"*.*")))):
        if not img_path.endswith(IMAGE_EXTENSION):
            continue
        x = to_tensor(read_image(img_path))
        if chop:
            patch_spliter = ImageSpliterTh(x, pch_size=chop_size, stride=chop_stride, sf=sf, extra_bs=1)
            for patch, index_infos in patch_spliter:
                patch_h, patch_w = patch.shape[2:]
                flag_pad = False
                if not (patch_h % 64 == 0 and patch_w % 64 == 0):
                    flag_pad = True
                    pad_h = (math.ceil(patch_h / 64)) * 64 - patch_h
                    pad_w = (math.ceil(patch_w / 64)) * 64 - patch_w
                    patch = F.pad(patch, pad=(0, pad_w, 0, pad_h), mode='reflect')
                pad_patch_h, pad_patch_w = patch.shape[2:]
                patch = F.interpolate(patch, size=(upscale*pad_patch_h, upscale*pad_patch_w), mode='bicubic')
                im_sr_pch = model.inference(patch)
                if flag_pad:
                    im_sr_pch = im_sr_pch[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(im_sr_pch, index_infos)
            out_img = patch_spliter.gather()
        else:
            x = F.interpolate(x, size=(img_size, img_size), mode='bicubic')
            out_img = model.inference(x)

        out_path = os.path.join(out_image_folder,os.path.basename(img_path))
        save_image(out_img, out_path)


    print("Done! images are at {}".format(out_image_folder))

if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Eval
    # ----------------------------
    run_infer(conf)