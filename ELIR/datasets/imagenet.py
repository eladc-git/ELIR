import torch
from torchvision.transforms import v2
from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
from ELIR.datasets.blindsr import degradation_realesrgan
from ELIR.datasets.realesrgan_dataset import RealESRGANDataset
import os
import glob


class ImagenetTrainDataset(Dataset):
    def __init__(self, image_folder, task, patch_size):
        super(ImagenetTrainDataset, self).__init__()
        self.images_path = self.get_file_path(image_folder, "ILSVRC2012_img_train")
        if task == 'bsr':
            opt = {'blur_kernel_size': 21,
                   'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                   'plateau_aniso'],
                   'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob': 0.1,
                   'blur_sigma': [0.2, 3.0], 'betag_range': [0.5, 4.0], 'betap_range': [1, 2.0],
                   'blur_kernel_size2': 15,
                   'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                    'plateau_aniso'],
                   'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], 'sinc_prob2': 0.1,
                   'blur_sigma2': [0.2, 1.5], 'betag_range2': [0.5, 4.0], 'betap_range2': [1, 2.0],
                   'final_sinc_prob': 0.8,
                   'use_hflip': True, 'use_rot': False}
            self.realESRGANDataset = RealESRGANDataset(self.images_path, opt)
        else:
            self.transform = self.preprocess()
        self.task = task
        self.patch_size = patch_size

    def get_file_path(self, image_folder, q):
        images_path = []
        image_folder_split = os.path.join(image_folder, q)
        for file in sorted(glob.glob(os.path.join(image_folder_split,"**/*.JPEG"))):
            images_path.append(file)
        return images_path

    def preprocess(self):
        transform = v2.Compose([
            v2.ToTensor()])
        return transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        data = self.realESRGANDataset[index]
        with torch.no_grad():
            lq, hq = degradation_realesrgan(data, patch_size=self.patch_size, sf=4, resize_back=True)
        return lq, hq



class Imagenet(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)
        task = dataset_params.get("task", "bsr")

        # Dataset
        dataset = ImagenetTrainDataset(path,
                                       task=task,
                                       patch_size=patch_size)
        # Loader
        loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)

        return loader
