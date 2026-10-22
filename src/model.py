import torch
import torch.nn as nn
from torchvision import models

class HybridModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        # Backbone ConvNext-Tiny
        weight_conv = (models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.convnext = models.convnext_tiny(weights=weight_conv)

        # Backbone Resnet101
        weights_res = (models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.resnet = models.resnet50(weights=weights_res)

        # Feature Extractor
        self.convnext.classifier = nn.Sequential(
            self.convnext.classifier[0],  # LayerNorm2d
            self.convnext.classifier[1]   # Flatten
        )
        self.resnet.fc = nn.Identity()

        # Feature Level Fusion
        convnext_dim = 768
        resnet_dim = 2048
        fused_dim = convnext_dim + resnet_dim

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f_conv = self.convnext(x)
        f_res = self.resnet(x)

        fused_features = torch.cat(
            [f_conv, f_res], dim=1
        )

        output = self.classifier(fused_features)
        return output