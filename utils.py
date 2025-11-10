import os
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def gpu_memory_report(prefix="GPU Memory"):
    """Print GPU memory usage report."""
    if not torch.cuda.is_available():
        print(f"{prefix}: CUDA not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1e9
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
    
    print(f"{prefix}:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Free:      {free:.2f} GB")


def save_checkpoint(model, optimizer, epoch, metrics, config, checkpoint_dir, best=False):
    """
    Save model checkpoint with standard taxonomy.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dict with train_loss, val_loss, val_accuracy, etc.
        config: Config object
        checkpoint_dir: Directory to save checkpoint
        best: If True, save as best_model.pt, else as epoch_N.pt
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
        'best_val_accuracy': metrics.get('best_val_accuracy', 0.0),
        'training_history': metrics.get('history', {}),
    }
    
    if best:
        path = checkpoint_dir / "best_model.pt"
    else:
        path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load checkpoint with standard taxonomy.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Dict with epoch, config, best_val_accuracy, training_history
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = {
        'epoch': checkpoint['epoch'],
        'config': checkpoint['config'],
        'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
        'training_history': checkpoint.get('training_history', {}),
    }
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {metadata['epoch']}")
    print(f"  Best val accuracy: {metadata['best_val_accuracy']:.4f}")
    
    return metadata


def cleanup_old_checkpoints(checkpoint_dir, keep_best=True, keep_last_n=3):
    """
    Delete old checkpoints, keeping best and last N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: If True, always keep best_model.pt
        keep_last_n: Number of most recent epoch_*.pt files to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    
    if len(epoch_files) > keep_last_n:
        files_to_delete = epoch_files[:-keep_last_n]
        for f in files_to_delete:
            f.unlink()
            print(f"Deleted old checkpoint: {f}")


def get_device():
    """Get cuda device if available, else cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    return total, trainable
