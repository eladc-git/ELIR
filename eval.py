from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
import pytorch_lightning as L
from ELIR.irsetup import IRSetup
import warnings
warnings.filterwarnings("ignore")


def run_eval(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Prepare datasets
    # ----------------------------
    dataset_cfg = conf.get("dataset_cfg")
    val_dataset = dataset_cfg.get('val_dataset')
    valloader = get_loader(val_dataset)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # ----------------------------
    # Evaluation
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    setup = IRSetup(model, eval_cfg=eval_cfg)
    trainer = L.Trainer(logger=False)
    set_seed(seed)
    results = trainer.validate(setup, dataloaders=valloader)
    metrics = eval_cfg.get("metrics")
    for metric in metrics:
        print("{}: {:0.4f}".format(metric, results[0][metric]), end =", ")

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
    run_eval(conf)