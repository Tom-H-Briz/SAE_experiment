import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from pathlib import Path
from tqdm import tqdm

from config import Config
from model import ViT
from data import get_dataloaders, inspect_batch
from utils import (
    set_seed, gpu_memory_report, save_checkpoint, load_checkpoint,
    cleanup_old_checkpoints, get_device, count_parameters
)


def create_optimizer(model, config):
    """Create Adam optimizer with weight decay."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    return optimizer


def create_lr_scheduler(optimizer, config, total_steps):
    """
    Create cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training config
        total_steps: Total training steps (epochs * batches_per_epoch)
    
    Returns:
        scheduler: LR scheduler
        warmup_steps: Number of warmup steps
    """
    warmup_steps = int(config.training.warmup_epochs * total_steps / config.training.num_epochs)
    
    # CosineAnnealingWarmRestarts with manual warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.num_epochs,
        T_mult=1,
        eta_min=1e-6,
    )
    
    return scheduler, warmup_steps


def warmup_lr(optimizer, step, warmup_steps, base_lr):
    """Apply linear warmup to learning rate."""
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_one_epoch(model, train_loader, optimizer, criterion, device, config, epoch, scaler=None):
    """
    Train for one epoch.
    
    Args:
        model: ViT model
        train_loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cuda/cpu)
        config: Config object
        epoch: Current epoch number
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        avg_loss: Average training loss
        avg_accuracy: Average training accuracy
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        if config.training.use_mixed_precision and scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_norm_grad_clip)
        
        # Optimizer step
        if config.training.use_mixed_precision and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}',
        })
        
        # Log memory periodically
        if (batch_idx + 1) % config.training.log_frequency == 0:
            gpu_memory_report(f"Epoch {epoch+1} Batch {batch_idx+1}")
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device, config, epoch):
    """
    Validate model on validation set.
    
    Args:
        model: ViT model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device
        config: Config object
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average validation loss
        avg_accuracy: Average validation accuracy
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if config.training.use_mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(images)
                loss = criterion(logits, labels)
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels)
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}',
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # Clear cache after validation
    torch.cuda.empty_cache()
    
    return avg_loss, avg_accuracy


def train_full(config=None, resume_from=None):
    """
    Full training pipeline.
    
    Args:
        config: Config object (uses default if None)
        resume_from: Path to checkpoint to resume from (optional)
    """
    if config is None:
        config = Config()
    
    print("\n" + "="*60)
    print("VISION TRANSFORMER TRAINING - MNIST")
    print("="*60 + "\n")
    
    # Setup
    set_seed(config.training.seed)
    device = get_device()
    
    # Create model
    print("\nCreating model...")
    model = ViT(config.model)
    model = model.to(device)
    count_parameters(model)
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(config)
    inspect_batch(train_loader, config)
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler, warmup_steps = create_lr_scheduler(optimizer, config, total_steps)
    
    # Mixed precision
    scaler = None
    if config.training.use_mixed_precision and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training (fp16)")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }
    
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        metadata = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = metadata['epoch'] + 1
        best_val_accuracy = metadata['best_val_accuracy']
        training_history = metadata['training_history']
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60 + "\n")
    
    gpu_memory_report("Initial")
    
    global_step = 0
    
    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch, scaler
        )
        
        # Update learning rate (warmup + cosine)
        if epoch < config.training.warmup_epochs:
            warmup_lr(optimizer, epoch, config.training.warmup_epochs, config.training.learning_rate)
        scheduler.step()
        
        # Validation
        if (epoch + 1) % config.training.validate_frequency == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, config, epoch)
        else:
            val_loss, val_acc = 0.0, 0.0
        
        # Track metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        if val_acc > 0:
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best checkpoint
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                metrics = {
                    'best_val_accuracy': best_val_accuracy,
                    'history': training_history,
                }
                save_checkpoint(
                    model, optimizer, epoch, metrics, config,
                    config.training.checkpoint_dir, best=True
                )
        
        # Save regular checkpoint
        metrics = {
            'best_val_accuracy': best_val_accuracy,
            'history': training_history,
        }
        save_checkpoint(
            model, optimizer, epoch, metrics, config,
            config.training.checkpoint_dir, best=False
        )
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(
            config.training.checkpoint_dir,
            keep_best=True,
            keep_last_n=config.training.keep_last_n
        )
        
        gpu_memory_report(f"End of Epoch {epoch+1}")
        
        # Early stopping
        if config.training.early_stopping_patience is not None:
            # TODO: implement early stopping logic
            pass
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    
    # Final memory report
    gpu_memory_report("Final")
    
    return model, training_history


if __name__ == "__main__":
    config = Config()
    model, history = train_full(config)
