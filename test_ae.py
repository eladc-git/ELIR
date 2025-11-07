from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision.io import read_image
import torch
import os
from ELIR.models.load_model import get_model
from tqdm import tqdm
from utils import get_device
import glob
import warnings
warnings.filterwarnings("ignore")
device = get_device()

IMAGE_EXTENSION = ('jpg','png','jpeg')


def preprocess(img_tensor, img_size):
    img_tensor = img_tensor.unsqueeze(0) / 255.0
    img_tensor = F.interpolate(img_tensor, size=(img_size, img_size), mode='bicubic')
    img_tensor = img_tensor.to(device)
    return img_tensor


def run_vae(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Create model
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # ----------------------------
    # Infer all images in folder
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    image_folder = eval_cfg.get("image_folder",[])
    out_image_folder = os.path.join(image_folder,"..","out")
    img_size = eval_cfg.get("image_size",512)
    os.makedirs(out_image_folder, exist_ok=True) # run folder
    for img_path in tqdm(sorted(glob.glob(os.path.join(image_folder,"*")))):
        if not img_path.endswith(IMAGE_EXTENSION):
            continue
        img_tensor = read_image(img_path)
        x = preprocess(img_tensor, img_size)
        y = model.decoder(model.encoder(x))
        out_path = os.path.join(out_image_folder,os.path.basename(img_path))
        save_image(y, out_path)
        print("MSE={:0.3f}",10*torch.log10(torch.mean((y-x.to(x.device))**2)))
        del img_tensor, y

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
    run_vae(conf)