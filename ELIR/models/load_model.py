from ELIR.utils import get_model_size
from safetensors import safe_open
from utils import get_device

device = get_device()


def get_model(cfg):
    model_name = cfg.get("name")
    model_path = cfg.get("path")
    model_params = cfg.get("params", {})
    trainable = cfg.get("trainable",False)

    if model_name == "elir":
        from ELIR.models.elir import Elir
        model = Elir(**model_params)
        model.load_weights(model_path)
        return model
    elif model_name == "lunet":
        from ELIR.models.lunet import LUnet
        model = LUnet(**model_params)
        model.load_weights(model_path)
    elif model_name == "rrdbnet":
        from ELIR.models.rrdbnet import RRDBNet
        model = RRDBNet(**model_params)
        model.load_weights(model_path)
    elif model_name == "tiny_enc":
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        from ELIR.models.taesd import TAESD
        model = TAESD()
        model.load_state_dict(pretrained.state_dict())
        model = model.encoder
    elif model_name == "tiny_dec":
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        from ELIR.models.taesd import TAESD
        model = TAESD()
        model.load_state_dict(pretrained.state_dict())
        model = model.decoder
    elif model_name == "taesd":
        from ELIR.models.taesd import TAESD
        model = TAESD()
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        model.load_state_dict(pretrained.state_dict())
    else:
        raise Exception("Model {} is unknown!".format(model_name))

    if trainable:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        model.train()
    else:
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    model.to(device)
    print("{} was created! Number of parameters: {:0.2f}M".format(model_name, get_model_size(model)/1e6))
    return model



