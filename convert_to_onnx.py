import torch
import torch.nn as nn
import torch.onnx

# Define your PyTorch model
class ASLModel(nn.Module):
    def __init__(self, num_classes):
        super(ASLModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the trained model
num_classes = 29  # Adjust based on your dataset
model = ASLModel(num_classes)
model.load_state_dict(torch.load('asl_model_state_dict.pt'))
model.eval()

# Create a dummy input matching the model's expected input shape
dummy_input = torch.randn(1, 3, 64, 64)

# Export to ONNX
torch.onnx.export(model, dummy_input, "asl_model.onnx", input_names=["input"], output_names=["output"], opset_version=13)

print("ONNX model exported successfully as 'asl_model.onnx'")
