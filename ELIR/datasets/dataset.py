from abc import abstractmethod
import torch
from PIL import Image, ImageOps
import numpy as np




def get_loader(ds_params):
    ds_name = ds_params.get("name")
    if ds_name == "CelebA":
        from ELIR.datasets.celeba import CelebA
        dl = CelebA().create_loaders(ds_params)
    elif ds_name == "FFHQ":
        from ELIR.datasets.ffhq import FFHQ
        dl = FFHQ().create_loaders(ds_params)
    elif ds_name == "CelebAdult":
        from ELIR.datasets.celebadult import CelebAdult
        dl = CelebAdult().create_loaders(ds_params)
    elif ds_name == "WebPhoto":
        from ELIR.datasets.webphoto import WebPhoto
        dl = WebPhoto().create_loaders(ds_params)
    elif ds_name == "Imagenet":
        from ELIR.datasets.imagenet import Imagenet
        dl = Imagenet().create_loaders(ds_params)
    elif ds_name == "Imagenet256":
        from ELIR.datasets.imagenet256 import Imagenet256
        dl = Imagenet256().create_loaders(ds_params)
    else:
        raise Exception("Dataset is unknown!")

    print("{} dataset was loaded! Number of samples: {}".format(ds_name, len(dl.dataset)))
    return dl

class BasicLoader(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_loaders(self, dataset_params):
        raise NotImplemented(f'{self.__class__.__name__} have to be implemented!')

    def get_labels(self, dataset):
        labels = []
        for _,y in dataset:
            labels.append(y)
        return labels

    def get_mean_std(self, dataset):
        sum1, sum2 = 0, 0
        for img,_ in dataset:
            sum1 += torch.mean(img, dim=(0,2,3))
            sum2 += torch.mean(img ** 2, dim=[0,2,3])
        count = len(dataset)
        mean = sum1 / count
        var = (sum2 / count) - (mean ** 2)
        std = torch.sqrt(var)
        return mean, std

def aug(img: Image, patch_size, crop=True) -> Image:
    # Random Crop patch size
    if crop:
        W, H = img.size
        hcrop, wcrop = np.random.randint(max(1, H - patch_size)), np.random.randint(max(1, W - patch_size))
        img = img.crop((wcrop, hcrop, min(W, wcrop + patch_size), min(H, hcrop + patch_size)))
    # Random flip
    if torch.rand(1) < 0.5:
        img = ImageOps.mirror(img)
    return img

class Padding2Multiple(object):
    def __init__(self, pad_to_multiple=8):
        self.pad_to_multiple = pad_to_multiple

    def get_pad(self, x):
        return (x // self.pad_to_multiple + int(x % self.pad_to_multiple != 0)) * self.pad_to_multiple - x

    def __call__(self, img):
        W, H = img.size
        if W % self.pad_to_multiple == 0 and H % self.pad_to_multiple == 0:
            return img
        pad_h, pad_w = self.get_pad(H), self.get_pad(W)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class ResizeLongEdge(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if h == w:
            new_h, new_w = self.size, self.size
            pad_h, pad_w =0, 0
        elif h < w:
            new_h, new_w = int(h * (self.size / w)), self.size
            pad_h, pad_w = self.size - new_h, 0
        else:
            new_h, new_w = self.size, int(w * (self.size / h))
            pad_h, pad_w = 0, self.size - new_w

        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class Padding2Size(object):
    def __init__(self, H_target, W_target):
        self.H_target = H_target
        self.W_target = W_target

    def __call__(self, img):
        W, H = img.size
        pad_h, pad_w = max(0,self.H_target-H), max(0,self.W_target-W)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class AddGaussianNoise(object):
    def __init__(self, std_high=1, std_low=0):
        self.std_high = std_high
        self.std_low = std_low

    def __call__(self, img):
        std = np.random.uniform(low=self.std_low, high=self.std_high)
        img = np.asarray(img)
        img = img + np.random.randn(*img.shape) * std
        img = img.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class MaskInpaint(object):
    def __init__(self, prob=0.9):
        self.prob = prob

    def __call__(self, img):
        x = np.asarray(img)
        total = x.shape[0] * x.shape[1]
        mask_vec = np.ones([1, x.shape[0] * x.shape[1]])
        samples = np.random.choice(x.shape[0] * x.shape[1], int(total * self.prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.reshape((x.shape[0], x.shape[1],1))
        mask_b = np.repeat(mask_b, 3, axis=2)
        mask = np.ones_like(mask_b)
        mask[:, ...] = mask_b
        y = x * mask
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class AddColorization(object):
    def __init__(self, std_high=1, std_low=0):
        self.std_high = std_high
        self.std_low = std_low

    def __call__(self, img):
        x = np.asarray(img)
        # RGB to gray
        y = np.mean(x, axis=-1, keepdims=True)
        # Add noise
        std = np.random.uniform(low=self.std_low, high=self.std_high)
        y = y + np.random.randn(*y.shape) * std
        # Back to 3 dimesions
        y = np.repeat(y, 3, axis=-1)
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class MaskSqaure(object):
    def __init__(self, h=0.25, w=0.25):
        self.h = h
        self.w = w

    def __call__(self, img):
        x = np.asarray(img)
        y_max, x_max = x.shape[0], x.shape[1]
        hp, wp = int(x.shape[0] * self.h), int(x.shape[1] * self.w)
        ycp = np.random.randint(low=0, high=y_max-hp, size=(1,))
        xcp = np.random.randint(low=0, high=x_max-wp, size=(1,))
        y_start, y_stop = int(ycp), int(ycp+hp)
        x_start, x_stop = int(xcp), int(xcp+wp)
        mask = np.ones_like(x)
        mask[y_start:y_stop:,x_start:x_stop,:] = 0
        y = x*mask
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)



