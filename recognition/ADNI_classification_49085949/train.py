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
import warnings
warnings.filterwarnings("ignore", message="Overwriting .* in registry")

torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for speed

# ----------------------------
# Model report
# ----------------------------
def model_report(model, input_size=(1,224,224)):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    try:
        flops, _ = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
        print(f"FLOPs: {flops}")
    except Exception as e:
        print("FLOPs calculation failed.", str(e))

# ----------------------------
# Training function with OneCycleLR
# ----------------------------
def train_amp(model, train_loader, model_name='small-in22k',
              device='cuda', epochs=100, base_lr=1e-3, weight_decay=0.05,
              results_dir='results', early_stopping=True, patience=15):

    device = torch.device(device)
    use_cuda = (device.type == 'cuda') and torch.cuda.is_available()
    scaler = torch.amp.GradScaler() if use_cuda else None
    autocast_ctx = lambda: torch.amp.autocast(device_type='cuda') if use_cuda else nullcontext()

    # Prepare directories
    os.makedirs(os.path.join(results_dir, 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # OneCycleLR
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=base_lr * 3,          # peak lr
        total_steps=total_steps,
        pct_start=0.2,               # 前20% steps升高 lr
        anneal_strategy='cos',
        div_factor=25,               # 初始 lr = max_lr/div_factor
        final_div_factor=1e4         # 最终 lr = max_lr/final_div_factor
    )

    best_train_acc, best_epoch, no_improve_epochs = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)

        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step() 

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix({
                "Loss": f"{running_loss/total:.4f}",
                "Acc": f"{correct/total:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        train_loss, train_acc = running_loss / total, correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if train_acc > best_train_acc:
            best_train_acc, best_epoch = train_acc, epoch+1
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if early_stopping and no_improve_epochs >= patience and best_train_acc > 0.8:
            print(f"Early stopping at epoch {epoch+1}. Best Train Acc: {best_train_acc:.4f} (Epoch {best_epoch})")
            break

    # Save best model
    best_path = os.path.join(results_dir, f'{model_name}_best.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': best_train_acc,
    }, best_path)
    print(f"Best model saved at epoch {best_epoch} with Train Acc: {best_train_acc:.4f}")

    # Training
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # Plot Train Loss
    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(results_dir, 'plots', f'{model_name}_train_loss.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"Training loss curve saved to {loss_path}")

    # Plot Train Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(results_dir, 'plots', f'{model_name}_train_acc.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Training accuracy curve saved to {acc_path}")

    return model, history, best_path

# ----------------------------
# Evaluate test set
# ----------------------------
def evaluate_on_test(model, test_loader, checkpoint_path, device, save_dir):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = total_loss / total
    test_acc = total_correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    report_text = classification_report(all_labels, all_preds, target_names=['CN', 'AD'], digits=4, zero_division=0)
    print('Classification on test set:')
    print(report_text)
    with open(os.path.join(save_dir, 'classification_report_test.txt'), 'w') as f:
        f.write(report_text)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN','AD'], yticklabels=['CN','AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix/confusion_matrix_test.png'))
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt on AD/NC dataset")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--datapath', type=str, default=r"D:\MachineLearningData\AD_NC")
    parser.add_argument('--model_name', type=str, default="small-in22k")
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    results_root = r"recognition\ADNI_classification_49085949\results"
    os.makedirs(results_root, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading data from {args.datapath}...")
    loader = CustomDataLoader(datapath=args.datapath, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    loader.load_data()
    train_loader, test_loader = loader.get_loaders()
    meta = loader.get_meta()
    print(f"Data loaded: train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Dataset info: {meta}")

    print(f"Initializing ConvNeXt model ({args.model_name})...")
    model = convnext_small(in_chans=1, num_classes=2, pretrained=False).to(device)
    model_report(model)

    print("Starting training...")
    model, history, best_path = train_amp(model, train_loader,
                                          model_name=args.model_name,
                                          device=device,
                                          epochs=args.epochs,
                                          base_lr=args.lr,
                                          results_dir=results_root,
                                          early_stopping=args.early_stopping,
                                          patience=args.patience)
    print("Training completed. Evaluating on test set...")
    evaluate_on_test(model, test_loader, best_path, device, results_root)

if __name__ == "__main__":
    main()
