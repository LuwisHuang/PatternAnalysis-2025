import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
from torchvision import transforms
import glob
from PIL import Image
import random

# Custom Dataset for AD/NC images
class CustomImageDataset(Dataset):
    """Load images from folders automatically and assign labels"""

    def __init__(self, folder_paths, labels, transform=None):
        """
        Args:
            folder_paths (list[str]): List of folder paths for each class
            labels (list[int]): Corresponding class labels
            transform (callable, optional): Transformations to apply to images
        """
        self.img_paths = []  # list of all image file paths
        self.labels = []     # list of labels corresponding to images
        self.transform = transform

        # Iterate through each class folder and gather image paths
        for folder, label in zip(folder_paths, labels):
            # Grab all files in the folder
            files = glob.glob(os.path.join(folder, '*.*'))
            # Filter for supported image formats
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.img_paths.extend(files)
            self.labels.extend([label] * len(files))  # assign label to all images

    def __len__(self):
        """Return the total number of images"""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Return image and label at index `idx`"""
        img = Image.open(self.img_paths[idx]).convert('L')  # convert to grayscale
        if self.transform:
            img = self.transform(img)  # apply transformations if provided
        label = self.labels[idx]
        return img, label



class DataLoader: # DataLoader class for ADNI dataset
    """
    Optimized DataLoader for AD/NC dataset.
    Features:
        - Automatic train/val/test split
        - Data augmentation for training
        - Grayscale conversion
        - Normalization using dataset mean/std
        - Single image inference
    """

    def __init__(self, datapath=r"Data\AD_NC",
                 batch_size=64,
                 img_size=None,
                 split_ratio=(0.7, 0.15, 0.15),
                 seed=42):
        self.datapath = datapath
        self.batch_size = batch_size
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.seed = seed

        # PyTorch DataLoader placeholders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Dataset statistics
        self.mean = 0.0
        self.std = 0.0
        self.total_images = 0
        self.n_classes = 0

        # Ensure reproducibility
        torch.manual_seed(self.seed)

    def load_data(self):
        """Load dataset, compute mean/std, and create DataLoaders"""
        classes = ['AD', 'NC']  # define class names
        class_paths = [os.path.join(self.datapath, 'train', c) for c in classes]
        label_ids = list(range(len(classes)))

        # Compute dataset mean/std
        init_dataset = CustomImageDataset(class_paths, label_ids, transform=transforms.ToTensor())
        loader = TorchDataLoader(init_dataset, batch_size=self.batch_size, shuffle=False)

        mean = 0.0
        std = 0.0
        total_images = 0
        for imgs, _ in loader:
            batch = imgs.view(imgs.size(0), imgs.size(1), -1)  # flatten each image
            mean += batch.mean(2).sum(0)# sum mean of each batch
            std += batch.std(2).sum(0)# sum std of each batch
            total_images += imgs.size(0)

        mean /= total_images
        std /= total_images

        self.mean = mean.item()
        self.std = std.item()
        self.total_images = total_images
        self.n_classes = len(classes)

        # Determine image size
        if self.img_size is None:
            sample_img, _ = next(iter(loader))
            self.img_size = min(sample_img.shape[-2:])# use smallest dimension

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),# resize
            transforms.RandomHorizontalFlip(),# horizontal flip
            transforms.RandomVerticalFlip(p=0.5),# vertical flip
            transforms.RandomRotation(30),# rotation ±30°
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                   saturation=0.2, hue=0.1),# brightness/contrast/saturation/hue jitter
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # translation
            transforms.RandomCrop(self.img_size, padding=8, padding_mode='reflect'), # crop with padding
            transforms.ToTensor(),# convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std)# normalize
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),# resize
            transforms.ToTensor(),# convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std)# normalize
        ])

        # Load datasets
        train_dataset = CustomImageDataset(class_paths, label_ids, transform=train_transform)
        test_class_paths = [os.path.join(self.datapath, 'test', c) for c in classes]
        test_dataset = CustomImageDataset(test_class_paths, label_ids, transform=val_transform)

        # Concatenate train and test for splitting
        full_dataset = train_dataset + test_dataset
        total_len = len(full_dataset)
        train_len = int(total_len * self.split_ratio[0])
        val_len = int(total_len * self.split_ratio[1])
        test_len = total_len - train_len - val_len

        # Split dataset
        train_data, val_data, test_data = random_split(full_dataset, [train_len, val_len, test_len],
                                                       generator=torch.Generator().manual_seed(self.seed))

        # Ensure val/test uses only normalization (no augmentation)
        val_data.dataset.transform = val_transform
        test_data.dataset.transform = val_transform

        # Create PyTorch DataLoaders
        self.train_loader = TorchDataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = TorchDataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = TorchDataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    def transform_val_from_folder(self, folder_path):
        """Randomly read one image from folder and apply validation transforms"""
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) == 0:
            raise FileNotFoundError(f"No image in {folder_path}")

        img_path = os.path.join(folder_path, random.choice(files))
        img = Image.open(img_path).convert('L')

        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # resize
            transforms.ToTensor(),                              # convert to tensor
            transforms.Normalize(mean=self.mean, std=self.std)  # normalize
        ])
        return val_transform(img).unsqueeze(0)  # add batch dimension

    # Getters for DataLoaders and metadata
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_meta(self):
        """Return dataset meta information"""
        return {
            'total_images': self.total_images,
            'mean': self.mean,
            'std': self.std,
            'img_size': self.img_size,
            'channels': 1,
            'n_classes': self.n_classes
        }



# Example usage
# loader = DataLoader(datapath=r"D:\MachineLearningData\AD_NC", batch_size=64, img_size=224)
# loader.load_data()

# train_loader, val_loader, test_loader = loader.get_loaders()
# print(loader.get_meta())

# img_tensor = loader.transform_val_from_folder(r"D:\MachineLearningData\AD_NC\test\AD")
# print(img_tensor.shape)
