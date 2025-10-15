import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
import glob
import random

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
# Subset wrapper with transform
# ----------------------------
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ----------------------------
# Optimized DataLoader
# ----------------------------
class DataLoader:
    """
    Optimized AD/NC DataLoader with caching and data augmentation
    """
    def __init__(self, datapath=r"Data\AD_NC",
                 batch_size=32,
                 img_size=224,
                 split_ratio=(0.7,0.15,0.15),
                 seed=42,
                 num_workers=0,
                 pin_memory=False,
                 cache_in_memory=True):
        self.datapath = datapath
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_in_memory = cache_in_memory

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.mean = 0.0
        self.std = 1.0
        self.n_classes = 0
        self.total_images = 0

        torch.manual_seed(seed)

    def load_data(self):
        classes = ['AD','NC']
        label_ids = list(range(len(classes)))
        merged_datasets = []

        # Load images
        for c,label in zip(classes, label_ids):
            folder_train = os.path.join(self.datapath, 'train', c)
            folder_test = os.path.join(self.datapath, 'test', c)
            dataset = CachedImageDataset([folder_train, folder_test],
                                         [label, label],
                                         transform=None,
                                         cache_in_memory=self.cache_in_memory)
            merged_datasets.append(dataset)

        full_dataset = ConcatDataset(merged_datasets)
        self.total_images = len(full_dataset)
        self.n_classes = len(classes)

        # ----------------------------
        # Estimate mean/std with subset (fast)
        # ----------------------------
        temp_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        loader_for_stats = TorchDataLoader(
            SubsetWithTransform(full_dataset, temp_transform),
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
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
            transforms.RandomCrop(self.img_size, padding=8, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # ----------------------------
        # Split dataset
        # ----------------------------
        total_len = len(full_dataset)
        train_len = int(total_len*self.split_ratio[0])
        val_len = int(total_len*self.split_ratio[1])
        test_len = total_len - train_len - val_len
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # ----------------------------
        # Wrap with transform
        # ----------------------------
        train_data = SubsetWithTransform(train_subset, train_transform)
        val_data = SubsetWithTransform(val_subset, val_transform)
        test_data = SubsetWithTransform(test_subset, val_transform)

        # ----------------------------
        # DataLoaders
        # ----------------------------
        self.train_loader = TorchDataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )
        self.val_loader = TorchDataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )
        self.test_loader = TorchDataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_meta(self):
        return {
            'total_images': self.total_images,
            'mean': self.mean,
            'std': self.std,
            'img_size': self.img_size,
            'channels': 1,
            'n_classes': self.n_classes
        }

    def transform_val_from_folder(self, folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(files)==0:
            raise FileNotFoundError(f"No image in {folder_path}")
        img_path = os.path.join(folder_path, random.choice(files))
        img = Image.open(img_path).convert('L')
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return val_transform(img).unsqueeze(0)
