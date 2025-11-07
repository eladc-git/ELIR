from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
import pytorch_lightning as L
import torch
from ELIR.metrics import MetricEval
from ELIR.training.losses import get_loss
from torchvision.utils import save_image
from ELIR.training.ema_timm import ModelEMA
import os
import torch.nn.functional as F
from ELIR.utils import ImageSpliterTh
import math



class IRSetup(L.LightningModule):
    def __init__(self, model, fm_cfg={}, optimizer=None, scheduler=None, tmodel=None,
                 ema_decay=None, eval_cfg=None, run_dir=None, save_images=True):
        super().__init__()
        self.model = model
        self.fm_cfg = fm_cfg
        self.eval_cfg = eval_cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = eval_cfg.get("metrics",[])
        self.tmodel = tmodel
        self.cuda_device = torch.device(torch.cuda.current_device())
        self.metric_evals = [MetricEval(metric, self.cuda_device, run_dir) for metric in self.metrics]
        self.train_loss = []
        self.ema = None
        if ema_decay:
            self.ema = ModelEMA(model, device=self.cuda_device, decay=ema_decay)
        self.samples_dir = None
        if run_dir and save_images:
            self.samples_dir = os.path.join(run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)  # run folder
            self.samples = []

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        # Loss function
        loss = get_loss(self.model, x_hq, x_lq, self.fm_cfg, self.tmodel)

        self.train_loss.append(loss)
        if batch_idx % 5:
            self.log("train_loss", torch.mean(torch.Tensor(self.train_loss)).item(), logger=True, prog_bar=True)
            self.train_loss.clear()
        return loss

    def compute_metrics(self, x_hq_hat, x_hq):
        for metric_eval in self.metric_evals:
            metric_eval.compute(x_hq_hat, x_hq)

    def save_samples(self, current_epoch):
        save_image(torch.concat(self.samples,dim=0), os.path.join(self.samples_dir,"epoch_"+str(current_epoch)+".png"))

    def infer(self, x):
        if self.ema:
            return self.ema.model.inference(x)
        else:
            return self.model.inference(x)

    def validation_step(self, batch, batch_idx):
        x_lq, y = batch
        chop = self.eval_cfg.get("chop", None)
        if chop:
            sf = chop.get("sf", 4)
            upscale = chop.get("upscale", 4)
            chop_size = chop.get("chop_size", 256)
            chop_stride = chop.get("chop_stride", 224)
            patch_spliter = ImageSpliterTh(x_lq, pch_size=chop_size, stride=chop_stride, sf=sf, extra_bs=1)
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
                y_hat = self.infer(patch)
                if flag_pad:
                    y_hat = y_hat[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(y_hat, index_infos)
            y_hat = patch_spliter.gather()
        else:
            y_hat = self.infer(x_lq)

        if self.current_epoch > 0 and batch_idx < 2 and self.samples_dir and torch.cuda.current_device()==0:
            self.samples.append(y_hat[:2,...]) # save 2 images
            if len(self.samples) == 2: # save 2 batches
                self.save_samples(self.current_epoch)
                self.samples.clear()
        self.compute_metrics(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('global_step', self.global_step)
        for metric_eval in self.metric_evals:
            result = metric_eval.get_final().item()
            self.log(metric_eval.metric, result, sync_dist=True, prog_bar=True)
        torch.cuda.empty_cache()

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            if hasattr(self.ema.model,"fmir"):
                checkpoint['state_dict_fmir'] = self.ema.model.fmir.state_dict()
            if hasattr(self.ema.model,"mmse"):
                checkpoint['state_dict_mmse'] = self.ema.model.mmse.state_dict()
            if hasattr(self.ema.model,"enc"):
                checkpoint['state_dict_enc'] = self.ema.model.enc.state_dict()
            if hasattr(self.ema.model,"dec"):
                checkpoint['state_dict_dec'] = self.ema.model.dec.state_dict()
        else:
            if hasattr(self.model,"fmir"):
                checkpoint['state_dict_fmir'] = self.model.fmir.state_dict()
            if hasattr(self.model,"mmse"):
                checkpoint['state_dict_mmse'] = self.model.mmse.state_dict()
            if hasattr(self.model,"enc"):
                checkpoint['state_dict_enc'] = self.model.enc.state_dict()
            if hasattr(self.model,"dec"):
                checkpoint['state_dict_dec'] = self.model.dec.state_dict()
        return checkpoint

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [self.scheduler]
