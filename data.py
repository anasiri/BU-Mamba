import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def collate_fn(batch):
    """Custom collate function to handle batches."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def get_transforms(train=True):
    """Return appropriate transformations for training or validation datasets."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.02, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


def get_data_loaders(args):
    """Set up and return data loaders for the training and validation datasets."""
    dataset = ImageFolder(args.data_path)
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    train_loaders = []
    val_loaders = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda x: collate_fn([(get_transforms(train=True)(item[0]), item[1]) for item in x])
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: collate_fn([(get_transforms(train=False)(item[0]), item[1]) for item in x])
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders
