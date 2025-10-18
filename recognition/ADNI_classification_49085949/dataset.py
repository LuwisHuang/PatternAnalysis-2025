'''
AD/NC Dataset DataLoader Script

This file defines a custom DataLoader for the AD/NC dataset.
Features:
- In-memory caching of images to speed up training
- Data augmentation for training set
- Patient-level splitting of test set into validation and test subsets
- Automatic calculation of dataset mean and std for normalization
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Subset
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

# ----------------------------
# Custom Dataset with in-memory caching
# ----------------------------
class CachedImageDataset(Dataset):
    """AD/NC Dataset with optional in-memory cache to speed up training"""
    def __init__(self, folder_paths, labels, transform=None, cache_in_memory=True):
        self.img_paths = []  # store image file paths
        self.labels = []     # store corresponding labels
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.cache = {}      # dictionary idx -> PIL.Image

        # ---- Load all images from folders ----
        for folder, label in zip(folder_paths, labels):
            files = glob.glob(os.path.join(folder, '*.*'))
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # filter image files
            self.img_paths.extend(files)  # add to list
            self.labels.extend([label] * len(files))  # repeat label for all files

    def __len__(self):
        return len(self.img_paths)  # return dataset size

    def __getitem__(self, idx):
        # ---- Load image, either from cache or disk ----
        if self.cache_in_memory and idx in self.cache:
            img = self.cache[idx]  # retrieve from cache
        else:
            img = Image.open(self.img_paths[idx]).convert('L')  # load as grayscale
            if self.cache_in_memory:
                self.cache[idx] = img  # store in cache

        label = self.labels[idx]

        # ---- Apply transforms if provided ----
        if self.transform:
            img = self.transform(img)

        return img, label  # return image and label

# ----------------------------
# Optimized DataLoader for AD/NC dataset
# ----------------------------
class DataLoader:
    """
    AD/NC DataLoader with:
    - caching
    - data augmentation
    - patient-based val/test splitting
    """
    def __init__(self, datapath=r"Data\AD_NC",
                 batch_size=32,
                 img_size=224,
                 seed=42,
                 num_workers=0,
                 pin_memory=False,
                 cache_in_memory=True,
                 val_ratio=0.5):
        self.datapath = datapath
        self.batch_size = batch_size
        self.img_size = img_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_in_memory = cache_in_memory
        self.val_ratio = val_ratio

        # ---- Placeholders for dataloaders ----
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.mean = 0.0
        self.std = 1.0
        self.n_classes = 0
        self.total_images = 0

        torch.manual_seed(seed)  # set seed for reproducibility

    # ---- Load dataset and create train/val/test splits ----
    def load_data(self):
        classes = ['AD','NC']
        label_ids = list(range(len(classes)))

        # ---- Load training dataset ----
        train_folders = [os.path.join(self.datapath, 'train', c) for c in classes]
        self.train_dataset = CachedImageDataset(train_folders, label_ids, transform=None, cache_in_memory=self.cache_in_memory)

        # ---- Load full test dataset ----
        test_folders = [os.path.join(self.datapath, 'test', c) for c in classes]
        full_test_dataset = CachedImageDataset(test_folders, label_ids, transform=None, cache_in_memory=self.cache_in_memory)

        # ---- Patient-based split for validation and test ----
        patient_to_indices = {}
        for idx, path in enumerate(full_test_dataset.img_paths):
            filename = os.path.basename(path)
            patient_id = filename.split('_')[0]  # extract patient ID
            if patient_id not in patient_to_indices:
                patient_to_indices[patient_id] = []
            patient_to_indices[patient_id].append(idx)  # map patient to image indices

        all_patient_ids = list(patient_to_indices.keys())
        strat_labels = [full_test_dataset.labels[patient_to_indices[pid][0]] for pid in all_patient_ids]

        val_patient_ids, test_patient_ids = train_test_split(
            all_patient_ids,
            test_size=1-self.val_ratio,
            random_state=self.seed,
            shuffle=True,
            stratify=strat_labels
        )

        val_indices, test_indices = [], []
        for pid in val_patient_ids:
            val_indices.extend(patient_to_indices[pid])
        for pid in test_patient_ids:
            test_indices.extend(patient_to_indices[pid])

        self.val_dataset = Subset(full_test_dataset, val_indices)  # wrap val set
        self.test_dataset = Subset(full_test_dataset, test_indices)  # wrap test set

        # ---- Compute mean and std from training set ----
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
            batch = imgs.view(imgs.size(0), imgs.size(1), -1)  # flatten HxW
            mean += batch.mean(2).sum(0)
            std += batch.std(2).sum(0)
            n_samples += imgs.size(0)
        self.mean = (mean / n_samples).item()  # dataset mean
        self.std = (std / n_samples).item()    # dataset std

        # ---- Define transforms ----
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # resize
            transforms.RandomHorizontalFlip(p=0.5),            # horizontal flip
            transforms.RandomRotation(10),                     # rotate ±10°
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),  # translate & scale
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # brightness/contrast jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)   # normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # ---- Assign transforms to datasets ----
        self.train_dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.test_transform
        self.test_dataset.dataset.transform = self.test_transform

        # ---- Update metadata ----
        self.total_images = len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        self.n_classes = len(classes)

        # ---- Create DataLoaders ----
        self.train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers>0 else False
        )
        self.val_loader = TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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

    # ---- Return train/val/test loaders ----
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    # ---- Return dataset meta info ----
    def get_meta(self):
        return {
            'total_images': self.total_images,
            'mean': self.mean,
            'std': self.std,
            'img_size': self.img_size,
            'channels': 1,
            'n_classes': self.n_classes
        }
