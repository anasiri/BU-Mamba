import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
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


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(args, seed):
    """Set up data loaders for training, validation, and testing."""
    set_seed(seed)
    dataset = ImageFolder(args.data_path)
    total_size = len(dataset)
    test_samples = int(total_size * args.test_size)
    val_samples = int(total_size * args.val_size)
    train_samples = total_size - test_samples - val_samples

    # Split dataset into train, validation, and test
    train_dataset, temp_dataset = random_split(dataset, [train_samples, total_size - train_samples],
                                               generator=torch.Generator().manual_seed(seed))
    val_dataset, test_dataset = random_split(temp_dataset, [val_samples, test_samples],
                                             generator=torch.Generator().manual_seed(seed))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=lambda x: collate_fn(
                                  [(get_transforms(train=True)(item[0]), item[1]) for item in x])
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=lambda x: collate_fn(
                                [(get_transforms(train=False)(item[0]), item[1]) for item in x])
                            )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=lambda x: collate_fn(
                                 [(get_transforms(train=False)(item[0]), item[1]) for item in x])
                             )

    return train_loader, val_loader, test_loader

# def get_data_loaders(args):
#     """Set up and return data loaders for the training and validation datasets."""
#     dataset = ImageFolder(args.data_path)
#
#     # Split the dataset into train+val and test sets
#     num_test_samples = int(len(dataset) * args.test_size)  # Define args.test_size in your arguments
#     num_train_val_samples = len(dataset) - num_test_samples
#     train_val_dataset, test_dataset = random_split(dataset, [num_train_val_samples, num_test_samples], generator=torch.Generator().manual_seed(args.seed))
#
#     kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
#
#     train_loaders = []
#     val_loaders = []
#
#     for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
#         train_subset = Subset(dataset, train_ids)
#         val_subset = Subset(dataset, val_ids)
#
#         train_loader = DataLoader(
#             dataset=train_subset,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=args.num_workers,
#             collate_fn=lambda x: collate_fn([(get_transforms(train=True)(item[0]), item[1]) for item in x])
#         )
#
#         val_loader = DataLoader(
#             dataset=val_subset,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers,
#             collate_fn=lambda x: collate_fn([(get_transforms(train=False)(item[0]), item[1]) for item in x])
#         )
#
#         train_loaders.append(train_loader)
#         val_loaders.append(val_loader)
#
#     return train_loaders, val_loaders
