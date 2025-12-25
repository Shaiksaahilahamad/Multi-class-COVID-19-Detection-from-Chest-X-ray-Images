import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.config import cfg
from src.model import build_model
from src.utils import ensure_dirs

def load_image(path: str, img_size: int):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    x = tfm(img).unsqueeze(0)
    return img, x

def overlay_heatmap(original_rgb, heatmap_2d):
    """
    original_rgb: PIL Image (RGB) resized to (img_size, img_size)
    heatmap_2d: 2D numpy array (Hc, Wc) values in [0,1]
    """
    original = np.array(original_rgb)  # (H, W, 3) RGB
    h, w = original.shape[:2]

    # Resize heatmap to match image size (VERY IMPORTANT)
    heatmap_resized = cv2.resize(heatmap_2d, (w, h))

    # Convert to 0-255 uint8 for applyColorMap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # (H, W, 3) BGR

    # Convert original RGB -> BGR for OpenCV
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    # Blend
    overlay_bgr = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    # Back to RGB for PIL saving
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb

def main(image_path: str):
    ensure_dirs(cfg.gradcam_dir)

    ckpt_path = cfg.model_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    classes = ckpt["classes"]

    model, target_layer_name = build_model(ckpt["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()

    # Get target layer (ResNet layer4, etc.)
    target_layer = dict([*model.named_modules()])[target_layer_name]

    activations = None
    gradients = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output

    def backward_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    original_img, x = load_image(image_path, ckpt["img_size"])
    x = x.to(cfg.device)

    logits = model(x)
    pred_idx = logits.argmax(dim=1).item()
    pred_name = classes[pred_idx]
    print("Prediction:", pred_name)

    score = logits[:, pred_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    # Grad-CAM
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])  # (C,)
    act = activations.squeeze(0)  # (C, H, W)
    for i in range(act.shape[0]):
        act[i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(act, dim=0)
    heatmap = F.relu(heatmap)
    heatmap /= (heatmap.max() + 1e-8)
    heatmap = heatmap.detach().cpu().numpy()

    overlay = overlay_heatmap(original_img.resize((ckpt["img_size"], ckpt["img_size"])), heatmap)

    out_path = cfg.gradcam_dir / f"gradcam_{Path(image_path).stem}_{pred_name}.png"
    Image.fromarray(overlay).save(out_path)
    print(f"✅ Saved Grad-CAM to: {out_path}")

    fh.remove()
    bh.remove()

if __name__ == "__main__":
    # Example:
    # python -m src.gradcam -- but simplest is edit below line to your image path
    sample = r"C:\Users\sksaa\OneDrive\Desktop\Multi-class COVID-19 Detection from Chest X-ray Images\data\test\Covid\094.png"
    main(sample)
