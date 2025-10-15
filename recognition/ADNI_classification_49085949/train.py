import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import convnext_small
from ptflops import get_model_complexity_info

import argparse
from dataset import DataLoader as CustomDataLoader

from packaging import version
from contextlib import nullcontext

import warnings
warnings.filterwarnings("ignore", message="Overwriting .* in registry")

torch.backends.cudnn.benchmark = True  # cudnn benchmark for speed

# ----------------------------
# Model report function
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
# Training function with AMP
# ----------------------------
def train_amp(model_name='small-in22k', train_loader=None, val_loader=None, 
              device='cuda', epochs=20, lr=1e-3, results_dir='results',
              early_stopping=True, patience=15):
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device(str(device))

    use_cuda = (device.type == 'cuda') and torch.cuda.is_available()
    torch_ver = version.parse(torch.__version__)
    if use_cuda:
        scaler = torch.amp.GradScaler('cuda')
        if torch_ver >= version.parse("2.0"):
            def autocast_ctx(): return torch.amp.autocast('cuda')
        else:
            def autocast_ctx(): return torch.cuda.amp.autocast()
    else:
        scaler = None
        autocast_ctx = lambda: nullcontext()

    os.makedirs(os.path.join(results_dir, 'confusion_matrix'), exist_ok=True)
    
    model = convnext_small(num_classes=2, in_chans=1, pretrained=False)
    model = model.to(device)
    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()} | AMP enabled: {use_cuda}")
    
    print(f"=== Model Report ===")
    model_report(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, train_accs, val_accs = [], [], []
    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast_ctx():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{correct/total:.4f}'})

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = correct / total if total > 0 else 0.0
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, f'{model_name}_best.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        scheduler.step()
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if early_stopping and no_improve_epochs >= patience:
            print(f"Early stopping triggered! ({patience} epochs without improvement)")
            break

    # Classification report & confusion matrix
    report = classification_report(all_labels, all_preds, target_names=['AD','NC'])
    print("=== Classification Report ===")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AD','NC'], yticklabels=['AD','NC'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')

    cm_path = os.path.join(results_dir, 'confusion_matrix', f'{model_name}_cm.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return model, train_losses, train_accs, val_accs, report, cm


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt on ADNI dataset")
    parser.add_argument('--batch_size', type=int, help='Batch size for training/validation')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Initial learning rate')
    parser.add_argument('--device', type=str, help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--datapath', type=str, help='Path to dataset folder')
    parser.add_argument('--save_dir', type=str, help='Directory to save results')
    parser.add_argument('--model_name', type=str, help='ConvNeXt variant name')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    datapath = args.datapath if args.datapath else r"D:\MachineLearningData\AD_NC"
    batch_size = args.batch_size if args.batch_size else 32
    epochs = args.epochs if args.epochs else 200
    lr = args.lr if args.lr else 5e-5
    save_dir = args.save_dir if args.save_dir else "results"
    model_name = args.model_name if args.model_name else "small-in22k"
    early_stopping = args.early_stopping if args.early_stopping else True
    patience = args.patience if args.patience else 15

    print(f"Loading data from {datapath}...")
    loader = CustomDataLoader(datapath=datapath, batch_size=batch_size, num_workers=4, pin_memory=True) 
    loader.load_data()
    train_loader, val_loader, test_loader = loader.get_loaders()
    meta = loader.get_meta()
    print(f"Data loaded: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"Dataset info: {meta}")

    print(f"Initializing ConvNeXt model ({model_name})...")
    model = convnext_small(in_chans=1, num_classes=2, pretrained=False)
    model_report(model)

    print("Starting training...")
    model, train_losses, train_accs, val_accs, report, cm = train_amp(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        results_dir=save_dir,
        early_stopping=early_stopping,
        patience=patience
    )

    print("Training completed.")
    print("=== Final Classification Report ===")
    print(report)
    print(f"Confusion matrix saved to: {os.path.join(save_dir, 'confusion_matrix', f'{model_name}_cm.png')}")

if __name__ == "__main__":
    main()
