import torch
import torch.nn as nn
import timm

class DeepfakeEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True):
        super(DeepfakeEfficientNet, self).__init__()
        # Load the pretrained backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace the final classifier head
        # Must match the architecture used during Colab training
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = DeepfakeEfficientNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, 1]
