import os
import shutil
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
import pyiqa



class MetricEval(object):
    def __init__(self, metric, device, out_dir=None):
        self.metric = metric
        self.values = []
        self.count = 0
        self.image_size = 512
        self.out_dir = out_dir
        if metric=="fid-g":
            # FID vs ground-truth
            self.evaluater = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            load_fid_statistics(os.path.join(os.path.dirname(__file__),"datasets","celeba_fid_stat.pt"), self.evaluater, device)
            self.evaluater.to(device)
        elif metric == "fid-f":
            # FID vs FFHQ dataset
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir,"out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
            self.evaluater = pyiqa.create_metric('fid', device=device)
        elif metric == "lpips":
            self.evaluater = pyiqa.create_metric(metric, device=device, net="vgg")
        elif metric == "save":
            # save images
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir,"out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            self.evaluater = pyiqa.create_metric(metric, device=device)

    def compute(self, x_hq_hat, x_hq=None):
        bs = x_hq_hat.shape[0]
        if self.metric=="fid-g":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            value = torch.zeros((len(x_hq_hat),))
            self.count += bs
        elif self.metric=="fid-f":
            self.image_size = x_hq.shape[-1]
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count)+".png"))
                self.count += 1
            value = torch.zeros((len(x_hq_hat),))
        elif self.metric in ["niqe","clipiqa","musiq"]:
            value = self.evaluater(x_hq_hat)
            self.count += bs
        elif self.metric == "save":
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count)+".png"))
                self.count += 1
            value = torch.zeros((len(x_hq_hat),))
        else:
            assert x_hq_hat.shape==x_hq.shape, f"Metric {self.metric} is expected to y and y_hat to be the same size! Make sure you use dataset with labels."
            value = self.evaluater(x_hq_hat, x_hq)
            self.count += bs
        self.values.append(value.reshape(-1))

    def get_final(self):
        if self.metric=="fid-g":
            final_value = self.evaluater.compute()
            self.evaluater.reset()
        elif self.metric=="fid-f":
            final_value = self.evaluater(self.out_dir,
                                         dataset_name="FFHQ",
                                         dataset_res=self.image_size,
                                         dataset_split="trainval70k",
                                         verbose=False)
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            final_value = torch.mean(torch.concat(self.values))
        self.count = 0
        self.values.clear()
        return final_value

def load_fid_statistics(stat_path, evaluator, device):
    real_features = torch.load(stat_path, map_location=device)
    evaluator.real_features_sum = real_features['real_features_sum']
    evaluator.real_features_cov_sum = real_features['real_features_cov_sum']
    evaluator.real_features_num_samples = real_features['real_features_num_samples']

def save_fid_statistics(evaluater):
    d = {"real_features_sum": evaluater.real_features_sum,
         "real_features_cov_sum": evaluater.real_features_cov_sum,
         "real_features_num_samples": evaluater.real_features_num_samples}
    torch.save(d, "a.pt")
