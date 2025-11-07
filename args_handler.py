import argparse


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml_path", '-y', type=str, required=True)
    parser.add_argument("--out_dir", "-o", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument("--epochs", '-e', type=int)
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=float, help="weight decay")
    parser.add_argument("--ema_decay", type=float, help="EMA decay")

    args_dict = vars(parser.parse_args())
    yaml_path = args_dict.pop("yaml_path")
    overides = {k: v for k, v in args_dict.items() if v is not None}
    return yaml_path, overides


def replace_key(conf, k, v):
    if k in conf:
        conf[k] = v
        return
    else:
        for sub_conf in conf:
            if isinstance(conf[sub_conf], dict):
                replace_key(conf[sub_conf], k, v)


def set_overides(conf, overides):
    for k,v in overides.items():
        replace_key(conf, k, v)

