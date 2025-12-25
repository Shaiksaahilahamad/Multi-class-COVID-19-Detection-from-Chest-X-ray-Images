import torch.nn as nn
from torchvision import models

def build_model(backbone: str, num_classes: int, pretrained: bool = True):
    backbone = backbone.lower().strip()

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer_name = "layer4"  # for Grad-CAM
        return model, target_layer_name

    if backbone == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer_name = "features"  # last conv block
        return model, target_layer_name

    if backbone == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        target_layer_name = "features"
        return model, target_layer_name

    raise ValueError(f"Unsupported backbone: {backbone}")
