from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data")
    train_dir: Path = Path("data/train")
    test_dir: Path = Path("data/test")
    model_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    plots_dir: Path = Path("outputs/plots")
    gradcam_dir: Path = Path("outputs/gradcam")

    # Training
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 10
    seed: int = 42

    # Model
    backbone: str = "resnet50"  # easy to swap later
    num_classes: int = 3
    label_smoothing: float = 0.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
