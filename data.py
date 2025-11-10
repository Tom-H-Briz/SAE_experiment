import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(config):
    """
    Create train and validation DataLoaders for MNIST.
    
    Args:
        config: Config object with training settings
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    
    # Normalization with MNIST standard values
    normalize = transforms.Normalize(
        mean=config.training.norm_mean,
        std=config.training.norm_std
    )
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Download and load MNIST
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,  # Drop incomplete final batch for consistent shapes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False,
    )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Normalization: mean={config.training.norm_mean}, std={config.training.norm_std}")
    
    return train_loader, val_loader


def inspect_batch(dataloader, config):
    """
    Grab one batch and inspect shapes/values for sanity checking.
    
    Args:
        dataloader: DataLoader to inspect
        config: Config object
    """
    images, labels = next(iter(dataloader))
    
    print(f"Batch inspection:")
    print(f"  Images shape: {images.shape}")
    print(f"  Images dtype: {images.dtype}")
    print(f"  Images min/max: {images.min():.4f} / {images.max():.4f}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels dtype: {labels.dtype}")
    print(f"  Unique labels: {torch.unique(labels).tolist()}")
    
    # Check normalization was applied correctly
    expected_mean = config.training.norm_mean
    expected_std = config.training.norm_std
    actual_mean = images.mean().item()
    actual_std = images.std().item()
    
    print(f"  Normalization check:")
    print(f"    Expected mean: {expected_mean}, Actual: {actual_mean:.4f}")
    print(f"    Expected std:  {expected_std}, Actual: {actual_std:.4f}")
