import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from torch.utils.data import DataLoader
from src.config import cfg
from src.dataset import get_datasets
from src.model import build_model
from src.utils import ensure_dirs


@torch.no_grad()
def main():
    ensure_dirs(cfg.plots_dir)

    ckpt = torch.load(cfg.model_dir / "best_model.pt", map_location=cfg.device)
    classes = ckpt["classes"]

    model, _ = build_model(ckpt["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()

    _, test_ds = get_datasets(cfg.train_dir, cfg.test_dir, ckpt["img_size"])
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    y_true = []
    y_prob = []

    for images, labels in loader:
        images = images.to(cfg.device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        y_prob.append(probs)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)            # (N,)
    y_prob = np.concatenate(y_prob, axis=0)    # (N, C)

    # One-vs-Rest ROC
    y_true_oh = label_binarize(y_true, classes=list(range(len(classes))))  # (N, C)

    plt.figure()
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()

    out_path = Path(cfg.plots_dir) / "roc_auc_curves.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"✅ Saved ROC-AUC plot to: {out_path}")


if __name__ == "__main__":
    main()
