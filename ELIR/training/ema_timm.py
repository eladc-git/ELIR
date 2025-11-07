import math
from copy import deepcopy
import torch
import torch.nn as nn


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, device, decay=0.999, updates=0):
        """
        Args:
            model, lrm (nn.Module): models to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.model = deepcopy(model.module if is_parallel(model) else model).eval().to(device)
        self.requires_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.requires_list.append(name)
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = (model.module.state_dict() if is_parallel(model) else model.state_dict())  # model state_dict
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point and k in self.requires_list:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()