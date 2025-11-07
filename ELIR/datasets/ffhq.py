import glob
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from ELIR.datasets.dataset import aug, BasicLoader, AddGaussianNoise, MaskInpaint, AddColorization, MaskSqaure
import os
from PIL import Image



class FFHQDataset(Dataset):
    def __init__(self, image_folder, task, patch_size=512):
        super(FFHQDataset, self).__init__()
        self.images_path = self.get_file_path(image_folder)
        self.transform_LQ, self.transform_HQ = self.preprocess(task, patch_size)
        self.task = task
        self.patch_size = patch_size

    def preprocess(self, task, patch_size):
        transform_HQ = v2.Compose([
            v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToTensor()])
        if task=="bfr":
            transform_LQ = v2.Compose([
                v2.Resize((patch_size,patch_size), interpolation=v2.InterpolationMode.BICUBIC),
                v2.GaussianBlur(kernel_size=41, sigma=(0.1, 15)),
                v2.RandomResize(min_size=patch_size//32, max_size=int(1.2*patch_size)),
                AddGaussianNoise(std_high=20),
                v2.JPEG(quality=(30,100)),
                v2.Resize((patch_size, patch_size)),
                v2.ToTensor()])
        elif task=="sr": # Super-Resolution
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

    def get_file_path(self, image_folder):
        images_path = []
        for file in sorted(glob.glob(os.path.join(image_folder,"**/*.png"))):
            images_path.append(os.path.join(image_folder, file))
        return images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        img = Image.open(img_path).convert("RGB")
        img = aug(img, self.patch_size, crop=False)
        hq = self.transform_HQ(img)
        lq = self.transform_LQ(img)
        return lq, hq


class FFHQ(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 16)
        patch_size = dataset_params.get("patch_size", 512)
        num_workers = dataset_params.get("num_workers", 4)
        task = dataset_params.get("task", "bfr")

        # Datasets
        dataset = FFHQDataset(path,
                              task=task,
                              patch_size=patch_size)

        # Loaders
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=True)

        return loader
