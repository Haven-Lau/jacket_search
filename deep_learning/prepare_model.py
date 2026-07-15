import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3SmallFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained mobilenet_v3_small with default weights (trained on ImageNet)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = models.mobilenet_v3_small(weights=weights)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 224, 224)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # shape: (batch_size, 576)
        
        # L2 Normalize the features: x = x / ||x||_2
        # This makes cosine similarity equivalent to simple dot product (matrix multiplication).
        eps = 1e-12
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
        x = x / norm
        return x

def main():
    # Force stdout and stderr to use UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    print("Preparing MobileNetV3-Small feature extractor...")
    model = MobileNetV3SmallFeatureExtractor()
    model.eval()
    
    # Create the output directory if it doesn't exist
    os.makedirs("deep_learning", exist_ok=True)
    onnx_path = os.path.join("deep_learning", "model.onnx")
    
    # Create dummy input for ONNX export (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    print("Model exported successfully!")
    
    # Simple check using the onnx library
    import onnx
    print("Loading exported ONNX model for validation...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker passed successfully!")

if __name__ == "__main__":
    main()
