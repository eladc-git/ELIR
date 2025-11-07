from torchvision.transforms import v2
from ELIR.datasets.dataset import BasicLoader
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import glob



class ImagenetValDataset(Dataset):
    def __init__(self, image_folder, patch_size):
        super(ImagenetValDataset, self).__init__()
        self.lq_images_path = self.get_file_path(image_folder, "lq")
        self.hq_images_path = self.get_file_path(image_folder, "gt")
        self.transform_LQ, self.transform_HQ = self.preprocess(patch_size)

    def get_file_path(self, image_folder, q):
        images_path = []
        image_folder_split = os.path.join(image_folder, q)
        for file in sorted(glob.glob(os.path.join(image_folder_split,"*.png"))):
            images_path.append(file)
        return images_path

    def __len__(self):
        return len(self.hq_images_path)

    def preprocess(self, patch_size):
        transform_LQ = v2.Compose([
            v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToTensor()])
        transform_HQ = v2.Compose([
            v2.ToTensor()])
        return transform_LQ, transform_HQ

    def __getitem__(self, index):
        img_path = self.lq_images_path[index]
        img = Image.open(img_path).convert("RGB")
        img_LQ = self.transform_LQ(img)
        img_path = self.hq_images_path[index]
        img = Image.open(img_path).convert("RGB")
        img_HQ = self.transform_HQ(img)
        return img_LQ, img_HQ



class Imagenet256(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)

        # Dataset
        dataset = ImagenetValDataset(path, patch_size)
        # Loader
        loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

        return loader
