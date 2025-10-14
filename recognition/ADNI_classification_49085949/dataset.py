import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
from torchvision import transforms
import glob


class CustomImageDataset(Dataset):
    """Custom Dataset to load images from specified folders and automatically assign labels"""
    def __init__(self, folder_paths, labels, transform=None):
        """
        Args:
            folder_paths (list[str]): folder path for each class
            labels (list[int]): label for each class
            transform (callable, optional): image transformations
        """
        self.img_paths = []
        self.labels = []
        self.transform = transform

        for folder, label in zip(folder_paths, labels):
            files = glob.glob(os.path.join(folder, '*.*'))  # get all files
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # filter image files
            self.img_paths.extend(files)
            self.labels.extend([label]*len(files))  # assign label to each image

    def __len__(self):
        return len(self.img_paths)  # return dataset size

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.img_paths[idx]).convert('L')  # open image and convert to grayscale
        if self.transform:
            img = self.transform(img)  # apply transforms
        label = self.labels[idx]
        return img, label  # return image and label


# Main DataLoader class
class DataLoader:
    """
    Optimized ADNI DataLoader
    - Supports train/val/test split
    - Supports augmentation, grayscale, normalization
    - Supports single image inference
    """
    def __init__(self, datapath=r"Data\AD_NC",
                 batch_size=64,
                 img_size=None,
                 split_ratio=(0.7,0.15,0.15),
                 seed=42):
        self.datapath = datapath
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.seed = seed

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # data statistics
        self.mean = 0.0
        self.std = 0.0
        self.total_images = 0
        self.n_classes = 0

        torch.manual_seed(self.seed)  # set random seed for reproducibility

    # -----------------------------
    # Step 1: Load dataset and compute mean/std
    # -----------------------------
    def load_data(self):
        classes = ['AD','NC']  # define class names
        class_paths = [os.path.join(self.datapath,'train', c) for c in classes]  # get train folders
        label_ids = list(range(len(classes)))  # assign numeric labels

        # initial dataset to compute mean/std
        init_dataset = CustomImageDataset(class_paths, label_ids, transform=transforms.ToTensor())
        loader = TorchDataLoader(init_dataset, batch_size=self.batch_size, shuffle=False)

        # compute mean and std
        mean = 0.0
        std = 0.0
        total_images = 0
        for imgs, _ in loader:
            batch = imgs.view(imgs.size(0), imgs.size(1), -1)  # flatten image
            mean += batch.mean(2).sum(0)  # accumulate batch mean
            std += batch.std(2).sum(0)    # accumulate batch std
            total_images += imgs.size(0)  # accumulate number of images
        mean /= total_images
        std /= total_images

        self.mean = mean.item()
        self.std = std.item()
        self.total_images = total_images
        self.n_classes = len(classes)

        # determine image size if not specified
        if self.img_size is None:
            sample_img, _ = next(iter(loader))
            self.img_size = min(sample_img.shape[-2:])  # take min(height, width)

        # -----------------------------
        # Define train and validation transforms
        # -----------------------------
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),  # resize to fixed size
            transforms.RandomHorizontalFlip(),                 # random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),             # random vertical flip
            transforms.RandomRotation(30),                    # random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # brightness/contrast/saturation/hue jitter
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),  # random translation
            transforms.RandomCrop(self.img_size, padding=8, padding_mode='reflect'),  # random crop with padding
            transforms.ToTensor(),                             # convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std) # normalize with computed mean/std
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),  # resize
            transforms.ToTensor(),                             # convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std) # normalize
        ])

        # Load full train and test datasets
        train_dataset = CustomImageDataset(class_paths, label_ids, transform=train_transform)
        test_class_paths = [os.path.join(self.datapath,'test', c) for c in classes]
        test_dataset = CustomImageDataset(test_class_paths, label_ids, transform=val_transform)

        # Split train/val/test
        full_dataset = train_dataset + test_dataset  # concatenate datasets
        total_len = len(full_dataset)
        train_len = int(total_len * self.split_ratio[0])
        val_len = int(total_len * self.split_ratio[1])
        test_len = total_len - train_len - val_len

        train_data, val_data, test_data = random_split(full_dataset, [train_len, val_len, test_len],
                                                       generator=torch.Generator().manual_seed(self.seed))  # reproducible split

        # set val/test transform to no augmentation
        val_data.dataset.transform = val_transform
        test_data.dataset.transform = val_transform


        # Create DataLoaders
        self.train_loader = TorchDataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = TorchDataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = TorchDataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    # Step 2: Transform single image for inference
    def transform_val_from_folder(self, folder_path):
        """
        Randomly read an image from a folder and apply inference transform
        """
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]  # list image files
        if len(files)==0:
            raise FileNotFoundError(f"No image in {folder_path}")

        from PIL import Image
        import random
        img_path = os.path.join(folder_path, random.choice(files))  # pick a random image
        img = Image.open(img_path).convert('L')  # convert to grayscale

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),  # resize
            transforms.ToTensor(),                             # convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std) # normalize
        ])
        return val_transform(img).unsqueeze(0)  # add batch dimension

    # Get DataLoaders and metadata
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


# =======================================================
# Example usage
# =======================================================
loader = DataLoader(datapath=r"D:\MachineLearningData\AD_NC", batch_size=64, img_size=224)
loader.load_data()

train_loader, val_loader, test_loader = loader.get_loaders()
print(loader.get_meta())

img_tensor = loader.transform_val_from_folder(r"D:\MachineLearningData\AD_NC\test\AD")
print(img_tensor.shape)
