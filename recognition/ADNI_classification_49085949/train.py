'''
AD/NC ConvNeXt Training and Evaluation Script

This script trains a ConvNeXt model on the AD/NC dataset with optional pretrained weights.
It uses:
- Automatic Mixed Precision (AMP) for faster training and reduced memory usage
- Gradient accumulation and gradient clipping
- ReduceLROnPlateau scheduler for dynamic learning rate adjustment
- Early stopping based on validation accuracy

Final outputs include:
- Best model checkpoint saved to results folder
- Training and validation loss/accuracy curves
- Classification report and confusion matrix on test set

'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import convnext_small
from ptflops import get_model_complexity_info
import argparse
from dataset import DataLoader as CustomDataLoader
from contextlib import nullcontext

# ---------------------------- GPU Optimization ----------------------------
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for faster training on fixed input sizes

# ---------------------------- Model Summary Function ----------------------------
def model_report(model, input_size=(1,224,224)):
    """
    Prints detailed model information:
    - Total parameters
    - Trainable parameters
    - FLOPs per forward pass (if ptflops is available)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")

    # Compute FLOPs using ptflops (optional)
    try:
        flops, _ = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
        print(f"FLOPs: {flops}")
    except Exception as e:
        print("FLOPs calculation failed.", str(e))

# ---------------------------- Training Function with AMP ----------------------------
def train_amp(model, train_loader, val_loader, model_name='small-in22k',
              device='cuda', epochs=120, base_lr=5e-4, weight_decay=0.05,
              results_dir='results', early_stopping=True, patience=15):

    device = torch.device(device)
    use_cuda = (device.type == 'cuda') and torch.cuda.is_available()

    # Gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler() if use_cuda else None
    autocast_ctx = lambda: torch.amp.autocast(device_type='cuda') if use_cuda else nullcontext()  # AMP context

    # ---------------------------- Create Results Directories ----------------------------
    os.makedirs(os.path.join(results_dir, 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # ---------------------------- Loss and Optimizer ----------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # CrossEntropy with label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)  # AdamW optimizer

    # Scheduler to reduce LR when val accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',       # monitor validation accuracy
        factor=0.5,       # LR reduction factor
        patience=5,       # wait epochs before reducing
        min_lr=1e-7,      # minimum LR
        verbose=True
    )

    accumulation_steps = 2  # Gradient accumulation steps
    best_val_acc, best_epoch, no_improve_epochs = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # ---------------------------- Epoch Loop ----------------------------
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)

        optimizer.zero_grad(set_to_none=True)  

        for batch_idx, (imgs, labels) in enumerate(train_bar):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # -- Forward and Backward with AMP --
            with autocast_ctx():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss for accumulation

            if scaler:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)  # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # -- Accumulate statistics --
            running_loss += loss.item() * imgs.size(0) * accumulation_steps
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # -- Update progress bar --
            train_bar.set_postfix({
                "Loss": f"{running_loss/total:.4f}",
                "Acc": f"{correct/total:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"  # Current learning rate
            })

        train_loss, train_acc = running_loss / total, correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # -- Validation Phase --
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_acc)  # Adjust LR based on validation accuracy

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ---------------------------- Early Stopping Logic ----------------------------
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch+1
            best_model_state = model.state_dict()  # Save best weights
            no_improve_epochs = 0
            print(f"New best validation accuracy: {best_val_acc:.4f}") 
        else:
            no_improve_epochs += 1
            print('no improve epoch:',no_improve_epochs)

        if early_stopping and no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
            break

        if best_val_acc > 0.8:
            print('Goal reached based on validation accuracy.')
            break

    # ---------------------------- Save Best Model ----------------------------
    best_path = os.path.join(results_dir, f'{model_name}_best.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': best_val_acc,
    }, best_path)
    print(f"Best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f}")

    # ---------------------------- Plot Training Curves ----------------------------
    # Loss Curve
    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label='Train Loss', color='red')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'plots', f'{model_name}_loss_curve.png'))
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8,5))
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'plots', f'{model_name}_acc_curve.png'))
    plt.close()

    # Learning Rate Curve
    plt.figure(figsize=(8,5))
    plt.plot(history['lr'], label='Learning Rate', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'plots', f'{model_name}_lr_curve.png'))
    plt.close()

    return model, history, best_path

# ---------------------------- Evaluate Dataset Function ----------------------------
def evaluate_dataset(model, loader, device, save_dir, name='test'):
    """
    Evaluate model performance and save results:
    - Loss and accuracy
    - Classification report
    - Confusion matrix heatmap
    """
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute final loss and accuracy
    loss = total_loss / total
    acc = total_correct / total
    print(f"{name.capitalize()} Loss: {loss:.4f}, {name.capitalize()} Acc: {acc:.4f}")

    # Classification report
    report_text = classification_report(all_labels, all_preds, target_names=['CN', 'AD'], digits=4, zero_division=0)
    print(f'Classification report on {name} set:')
    print(report_text)
    with open(os.path.join(save_dir, f'classification_report_{name}.txt'), 'w') as f:
        f.write(report_text)

    # Confusion matrix heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN','AD'], yticklabels=['CN','AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({name.capitalize()})')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix/confusion_matrix_{name}.png'))
    plt.close()

# ---------------------------- Main Script ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt on AD/NC dataset with validation and test evaluation")
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--lr', type=float, default=1e-4)  
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--datapath', type=str, default=r"D:\MachineLearningData\AD_NC")
    parser.add_argument('--model_name', type=str, default="small-in22k")
    parser.add_argument('--early_stopping', action='store_true',default=False)
    parser.add_argument('--patience', type=int, default=15)  
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.1)  
    args = parser.parse_args()

    # ---------------------------- Results Directory ----------------------------
    results_root = r"recognition\ADNI_classification_49085949\results"
    os.makedirs(results_root, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------------------------- Load Data ----------------------------
    print(f"Loading data from {args.datapath}...")
    loader = CustomDataLoader(datapath=args.datapath, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    loader.load_data()
    train_loader, val_loader, test_loader = loader.get_loaders()
    meta = loader.get_meta()
    print(f"Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Dataset info: {meta}")

    # ---------------------------- Initialize Model ----------------------------
    print(f"Initializing ConvNeXt model ({args.model_name})...")
    model = convnext_small(in_chans=1, num_classes=2, pretrained=args.pretrained).to(device)
    model_report(model)

    # ---------------------------- Train Model ----------------------------
    print("Starting training...")
    model, history, best_path = train_amp(model, train_loader, val_loader,
                                          model_name=args.model_name,
                                          device=device,
                                          epochs=args.epochs,
                                          base_lr=args.lr,
                                          weight_decay=args.weight_decay,
                                          results_dir=results_root,
                                          early_stopping=args.early_stopping,
                                          patience=args.patience)
    print("Training completed.")

    # ---------------------------- Evaluate on Test Set ----------------------------
    print("Evaluating on test set...")
    evaluate_dataset(model, test_loader, device, results_root, name='test')


if __name__ == "__main__":
    main()
