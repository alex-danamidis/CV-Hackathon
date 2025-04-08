import torch
import torch.onnx
import torch.nn as nn
from torchvision import datasets, transforms

# Define the model to match the architecture of the saved state dict
class ASLModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),  # Conv4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # FC layers (adjusted to match the state dict)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),  # Input size is adjusted for the final output of conv layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten the output for FC layers
        x = self.fc(x)
        return x

# Load the dataset to match the number of classes
dataset_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\ASL_Dataset\asl_alphabet_train\asl_alphabet_train'
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_data = datasets.ImageFolder(dataset_path, transform=transform)
num_classes = len(train_data.classes)  # Ensure this is 29 for ASL dataset

# Initialize the model
model = ASLModel(num_classes=num_classes)

# Load the saved state dict
state_dict_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\asl_model_state_dict.pt'
saved_state_dict = torch.load(state_dict_path)

# Check keys in the saved state dict
print("Keys in saved state dict:")
print(saved_state_dict.keys())

# Use strict=False to ignore the missing or unexpected keys
model.load_state_dict(saved_state_dict, strict=False)

# Ensure the model is in evaluation mode for inference
model.eval()

# Dummy input to test the model
dummy_input = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Export the model to ONNX format
onnx_file_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\asl_model.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12  # Ensure compatibility with ONNX opset version
)

print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
