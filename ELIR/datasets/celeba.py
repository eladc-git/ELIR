import glob
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import os
from ELIR.datasets.dataset import MaskInpaint, BasicLoader, AddGaussianNoise, AddColorization, MaskSqaure
from PIL import Image



class CelebADataset(Dataset):
    def __init__(self, image_folder, task, patch_size):
        super(CelebADataset, self).__init__()
        self.task = task
        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "hq")
        self.transform_LQ, self.transform_HQ = self.preprocess(task, patch_size)

    def preprocess(self, task, patch_size):
        transform_HQ = v2.Compose([
            v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToTensor()])
        if task=="bfr": # Blind Face Restoration
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToTensor()])
        elif task == "sr": # Super-Resolution
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                v2.Resize((patch_size // 8, patch_size // 8), interpolation=v2.InterpolationMode.BICUBIC),
                AddGaussianNoise(std_high=12.8, std_low=12.8),
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.NEAREST_EXACT),
                v2.ToTensor()])
        elif task == "denoising": # Image-Denoising
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                AddGaussianNoise(std_high=89.6, std_low=89.6),
                v2.ToTensor()])
        elif task == "inpainting":  # Inpainting
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                MaskInpaint(0.9),
                AddGaussianNoise(std_high=25.6, std_low=25.6),
                v2.ToTensor()])
        elif task == "colorization":  # Colorization
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                AddColorization(std_high=64, std_low=64),
                v2.ToTensor()])
        elif task == "mask":  # mask
            transform_LQ = v2.Compose([
                v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                MaskSqaure(0.5,0.5),
                AddGaussianNoise(std_high=25.6, std_low=25.6),
                v2.ToTensor()])
        else:
            assert False, "Task is not supported!"
        return transform_LQ, transform_HQ

    def get_file_path(self, image_folder, q):
        images_path = []
        for file in sorted(glob.glob(os.path.join(image_folder,q,"*.png"))):
            images_path.append(os.path.join(image_folder, file))
        return images_path

    def __len__(self):
        return len(self.hq_images_path)

    def __getitem__(self, index):
        img_HQ_path = self.hq_images_path[index]
        img_HQ = Image.open(img_HQ_path).convert("RGB")
        hq = self.transform_HQ(img_HQ)
        if self.task=="bfr":
            img_LQ_path = self.lq_images_path[index]
            img_LQ = Image.open(img_LQ_path).convert("RGB")
            lq = self.transform_LQ(img_LQ)
        else:
            lq = self.transform_LQ(img_HQ)
        return lq, hq


class CelebA(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        patch_size = dataset_params.get("patch_size", 512)
        num_workers = dataset_params.get("num_workers", 4)
        task = dataset_params.get("task", "bfr")

        # Datasets
        dataset = CelebADataset(os.path.join(path, "test"), task, patch_size)

        # Loaders
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)

        return loader
