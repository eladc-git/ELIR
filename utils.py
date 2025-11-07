import random
import numpy as np
import torch
import hashlib
import os



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

def set_train(model):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()

def set_eval(model):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

def get_hash(conf):
    return hashlib.md5(str(conf).encode('utf-8')).hexdigest()

