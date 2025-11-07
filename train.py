from args_handler import argument_handler, set_overides
import yaml
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
from ELIR.training.tparmas import get_opt_sched
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ELIR.irsetup import IRSetup
import os
import torch
import warnings
warnings.filterwarnings("ignore")



def run_train(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Save configuration
    # ----------------------------
    out_dir = env_cfg.get("out_dir")
    os.makedirs(out_dir, exist_ok=True) # out folder
    run_name = env_cfg.get("run_name")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True) # run folder
    conf_path = os.path.join(run_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.dump(conf, outfile)
    print("Configuration: {}", conf)

    # ----------------------------
    # Prepare datasets
    # ----------------------------
    dataset_cfg = conf.get("dataset_cfg")
    train_dataset = dataset_cfg.get('train_dataset')
    trainloader = get_loader(train_dataset)
    val_dataset = dataset_cfg.get('val_dataset')
    valloader = get_loader(val_dataset)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # Teacher model
    tmodel_cfg = model_cfg.get("teacher_cfg", None)
    tmodel = get_model(tmodel_cfg) if tmodel_cfg is not None else None

    # ----------------------------
    # Training
    # ----------------------------
    fm_cfg = conf.get("fm_cfg",{})
    train_cfg = conf.get("train_cfg")
    optimizer, scheduler = get_opt_sched(train_cfg, model)
    eval_cfg = conf.get("eval_cfg")

    # WandB
    wandbLogger = None
    if train_cfg.get("wandb",False):
        import wandb
        print("WandB is enable!")
        wandb.init(project=env_cfg.get("project_name"), dir=run_dir, group=run_name)
        wandbLogger = WandbLogger(project=env_cfg.get("project_name"), dir=run_dir)
        wandb.log(dict(**conf))

    # Training
    train_setup = IRSetup(model,
                         fm_cfg=fm_cfg,
                         tmodel=tmodel,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         ema_decay=train_cfg.get("ema_decay", 0.999),
                         eval_cfg=eval_cfg,
                         run_dir=run_dir,
                         save_images=train_cfg.get("save_images", True))
    checkpoint = ModelCheckpoint(run_dir,
                                 every_n_epochs=1,
                                 save_weights_only=train_cfg.get("save_weights_only", True), save_top_k=1,
                                 save_on_train_epoch_end=True, verbose=True)
    trainer = L.Trainer(max_epochs=train_cfg.get("epochs"),
                        default_root_dir = run_dir,
                        callbacks=checkpoint,
                        strategy = "ddp",
                        devices = "auto",
                        accelerator="gpu",
                        logger = wandbLogger,
                        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps",0),
                        check_val_every_n_epoch = train_cfg.get("check_val_every_n_epoch",1),
                        max_steps = train_cfg.get("max_steps", -1))
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    set_seed(seed)
    trainer.fit(train_setup, trainloader, valloader, ckpt_path=train_cfg.get("ckpt_path", None))

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = trainer.validate(dataloaders=valloader, ckpt_path="last")
    metrics = eval_cfg.get("metrics")
    for metric in metrics:
        metric_value = results[0][metric]
        print("{}: {:0.4f}".format(metric, metric_value), end =", ")
        if train_cfg.get("wandb", False):
            wandb.log({"final_"+metric: metric_value})

    # ----------------------------
    # Save model
    # ----------------------------
    def adjust_weights(sd):
        sd = dict((key.replace("model.",""), value) for (key, value) in sd.items())
        return sd

    # Save latest model
    train_setup.ema.model.collapse()
    state_dict = train_setup.ema.model.state_dict()
    state_dict = adjust_weights(state_dict)
    torch.save(state_dict, os.path.join(run_dir, "elir.pth"))



if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Train
    # ----------------------------
    run_train(conf)