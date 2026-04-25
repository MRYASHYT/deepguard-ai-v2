import torch
import torch.nn as nn
import timm

class DeepfakeViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(DeepfakeViT, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Replace the head
        num_features = self.backbone.num_features
        self.backbone.head = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = DeepfakeViT()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
