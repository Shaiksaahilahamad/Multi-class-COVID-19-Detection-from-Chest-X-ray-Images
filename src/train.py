from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import cfg
from src.dataset import get_datasets
from src.model import build_model
from src.utils import seed_everything, ensure_dirs

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def main():
    seed_everything(cfg.seed)
    ensure_dirs(cfg.model_dir, cfg.outputs_dir, cfg.plots_dir, cfg.gradcam_dir)

    # Load datasets
    train_ds, test_ds = get_datasets(cfg.train_dir, cfg.test_dir, cfg.img_size)
    class_names = train_ds.classes
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Build model
    model, _ = build_model(cfg.backbone, cfg.num_classes, pretrained=True)
    model = model.to(cfg.device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    best_path = cfg.model_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        va_loss, va_acc = validate(model, test_loader, criterion, cfg.device)

        print(f"Epoch {epoch}/{cfg.epochs} | "
              f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
              f"Test: loss={va_loss:.4f}, acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": class_names,
                "backbone": cfg.backbone,
                "img_size": cfg.img_size
            }, best_path)
            print(f"✅ Saved best model to: {best_path} (acc={best_acc:.4f})")

    print("Done. Best Test Acc:", best_acc)

if __name__ == "__main__":
    main()
