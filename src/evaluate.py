import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader
from src.config import cfg
from src.dataset import get_datasets
from src.model import build_model
from src.utils import ensure_dirs

@torch.no_grad()
def main():
    ensure_dirs(cfg.plots_dir)

    ckpt_path = cfg.model_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    classes = ckpt["classes"]

    model, _ = build_model(ckpt["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()

    _, test_ds = get_datasets(cfg.train_dir, cfg.test_dir, ckpt["img_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    all_preds, all_labels = [], []

    for images, labels in test_loader:
        images = images.to(cfg.device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    out_path = cfg.plots_dir / "confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"\n✅ Saved confusion matrix to: {out_path}")

if __name__ == "__main__":
    main()
