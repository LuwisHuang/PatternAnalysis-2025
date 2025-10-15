import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torchvision import transforms
from PIL import Image
import glob

# ----------------------------
# Custom Dataset with cache
# ----------------------------
class CachedImageDataset(Dataset):
    """AD/NC Dataset with in-memory cache to speed up training"""
    def __init__(self, folder_paths, labels, transform=None, cache_in_memory=True):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.cache = {}  # idx -> PIL.Image

        for folder, label in zip(folder_paths, labels):
            files = glob.glob(os.path.join(folder, '*.*'))
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.img_paths.extend(files)
            self.labels.extend([label] * len(files))


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cache:
            img = self.cache[idx]
        else:
            img = Image.open(self.img_paths[idx]).convert('L')
            if self.cache_in_memory:
                self.cache[idx] = img

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# ----------------------------
# Optimized DataLoader
# ----------------------------
class DataLoader:
    """
    AD/NC DataLoader with caching and data augmentation
    """
    def __init__(self, datapath=r"Data\AD_NC",
                 batch_size=32,
                 img_size=224,
                 seed=42,
                 num_workers=0,
                 pin_memory=False,
                 cache_in_memory=True):
        self.datapath = datapath
        self.batch_size = batch_size
        self.img_size = img_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_in_memory = cache_in_memory

        self.train_loader = None
        self.test_loader = None
        self.mean = 0.0
        self.std = 1.0
        self.n_classes = 0
        self.total_images = 0

        torch.manual_seed(seed)

    def load_data(self):
        classes = ['AD','NC']
        label_ids = list(range(len(classes)))

        # ----------------------------
        # Load training dataset
        # ----------------------------
        train_folders = [os.path.join(self.datapath, 'train', c) for c in classes]
        self.train_dataset = CachedImageDataset(train_folders, label_ids, transform=None, cache_in_memory=self.cache_in_memory)

        # ----------------------------
        # Load test dataset
        # ----------------------------
        test_folders = [os.path.join(self.datapath, 'test', c) for c in classes]
        self.test_dataset = CachedImageDataset(test_folders, label_ids, transform=None, cache_in_memory=self.cache_in_memory)

        self.total_images = len(self.train_dataset) + len(self.test_dataset)
        self.n_classes = len(classes)

        # ----------------------------
        # Estimate mean/std from training set
        # ----------------------------
        temp_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        loader_for_stats = TorchDataLoader(
            CachedImageDataset(train_folders, label_ids, transform=temp_transform, cache_in_memory=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        mean = 0.0
        std = 0.0
        n_samples = 0
        for imgs, _ in loader_for_stats:
            batch = imgs.view(imgs.size(0), imgs.size(1), -1)
            mean += batch.mean(2).sum(0)
            std += batch.std(2).sum(0)
            n_samples += imgs.size(0)
        self.mean = (mean / n_samples).item()
        self.std = (std / n_samples).item()

        # ----------------------------
        # Transforms
        # ----------------------------
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
            transforms.RandomCrop(self.img_size, padding=8, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # ----------------------------
        # Wrap datasets with transforms
        # ----------------------------
        self.train_dataset.transform = self.train_transform
        self.test_dataset.transform = self.test_transform

        # ----------------------------
        # DataLoaders
        # ----------------------------
        self.train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )
        self.test_loader = TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )

    def get_loaders(self):
        return self.train_loader, self.test_loader

    def get_meta(self):
        return {
            'total_images': self.total_images,
            'mean': self.mean,
            'std': self.std,
            'img_size': self.img_size,
            'channels': 1,
            'n_classes': self.n_classes
        }
