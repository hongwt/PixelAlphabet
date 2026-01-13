"""
Training Script for PixelNet

Train the character recognition model on 24x24 images.
"""
import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.model import create_model
from src.dataset import get_dataloader, label_to_char, char_to_label
from src.loss import FocalLoss, create_loss_function


def log_confusion_matrix(preds, targets, writer, epoch):
    """
    Log specific confusion cases (Q vs 0, 8 vs B) and specific matrix.
    """
    # Key indices
    try:
        idx_0 = char_to_label('0')
        idx_Q = char_to_label('Q')
        idx_8 = char_to_label('8')
        idx_B = char_to_label('B')
    except ValueError:
        return # Skip if labels not found

    confusion = {
        '0_as_Q': 0, 'Q_as_0': 0,
        '8_as_B': 0, 'B_as_8': 0
    }

    for p, t in zip(preds, targets):
        if t == idx_0 and p == idx_Q: confusion['0_as_Q'] += 1
        if t == idx_Q and p == idx_0: confusion['Q_as_0'] += 1
        if t == idx_8 and p == idx_B: confusion['8_as_B'] += 1
        if t == idx_B and p == idx_8: confusion['B_as_8'] += 1
    
    print(f"  [Confusion] Q->0: {confusion['Q_as_0']}, 0->Q: {confusion['0_as_Q']}")
    print(f"  [Confusion] 8->B: {confusion['8_as_B']}, B->8: {confusion['B_as_8']}")
    
    if writer:
        writer.add_scalars('Confusion/Q_0', {'Q_as_0': confusion['Q_as_0'], '0_as_Q': confusion['0_as_Q']}, epoch)
        writer.add_scalars('Confusion/8_B', {'8_as_B': confusion['8_as_B'], 'B_as_8': confusion['B_as_8']}, epoch)

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """
    Train for one epoch.
    
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            batch_acc = 100 * correct / total
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} Acc: {batch_acc:.2f}%")
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate the model.
    
    Returns:
        (avg_loss, accuracy, preds, targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_targets


def main(args):
    """Main training loop."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader = get_dataloader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        data_root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(num_classes=36, dropout_rate=args.dropout)
    model = model.to(device)
    
    # Loss function (Hardcoded to Combined Loss as per requirements)
    criterion = create_loss_function(
        'combined',
        use_focal=True,
        use_label_smoothing=True,
        smoothing=0.1,
        lambda_focal=0.5
    )
    print("Using loss function: Combined (Focal + Label Smoothing)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Log confusion
        log_confusion_matrix(val_preds, val_targets, writer, epoch)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
    
    # Final summary
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("="*60)
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PixelNet for character recognition')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
