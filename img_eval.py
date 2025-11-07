from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from torchvision.io import read_image
import os
import torch.nn.functional as F
from ELIR.metrics import MetricEval
from tqdm import tqdm
from utils import get_device
import glob
import warnings
warnings.filterwarnings("ignore")
device = get_device()

IMAGE_EXTENSION = ('jpg','png','jpeg')



def preprocess(img_tensor):
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
    # Eval metrics
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    metrics = eval_cfg.get("metrics")
    metric_evals = [MetricEval(metric, device, None) for metric in metrics]

    # ----------------------------
    # Eval all images in folder
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    in_folder = eval_cfg.get("in_folder",None)
    gt_folder = eval_cfg.get("gt_folder",None)
    gt_scale_factor = eval_cfg.get("gt_scale_factor",1)

    for img_path in tqdm(sorted(glob.glob(os.path.join(in_folder,"*")))):
        if not img_path.endswith(IMAGE_EXTENSION):
            continue
        # Restored image
        img_tensor = read_image(img_path)
        img_restored = preprocess(img_tensor)
        # HQ image
        img_hq = None
        if gt_folder is not None:
            basename = os.path.basename(img_path)
            img_path = os.path.join(gt_folder,basename)
            img_tensor = read_image(img_path)
            img_hq = preprocess(img_tensor)
            img_hq = F.interpolate(img_hq, scale_factor=gt_scale_factor, mode='bicubic')
        # Compute metrics
        for metric_eval in metric_evals:
            metric_eval.compute(img_restored, img_hq)


    # Print metrics
    for metric_eval in metric_evals:
        result = metric_eval.get_final().item()
        print("{}: {:0.4f}".format(metric_eval.metric, result), end=", ")


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