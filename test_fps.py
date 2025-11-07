from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from utils import get_device
from time import time
import torch
import warnings
warnings.filterwarnings("ignore")
device = get_device()


def run_test_fps(conf):
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
    torch.cuda.empty_cache()
    eval_cfg = conf.get("eval_cfg")
    img_size = eval_cfg.get("image_size",512)
    x = torch.randn((1, 3, img_size, img_size),device=device)
    n_images = 200

    # Warming up
    for _ in range(10):
        model.inference(x)

    #--------------------------------------------------------------
    start = time()
    #--------------------------------------------------------------
    for _ in range(n_images):
        model.inference(x)
    #--------------------------------------------------------------
    stop = time()
    #--------------------------------------------------------------

    fps = n_images / (stop - start)
    print("FPS={:0.2f}".format(fps))


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
    run_test_fps(conf)