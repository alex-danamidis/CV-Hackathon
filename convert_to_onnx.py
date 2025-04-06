import torch
import torch.onnx
import torch.nn as nn
from torchvision import datasets, transforms

# Define the model class (must match training model)
class ASLModel(nn.Module):
    def __init__(self, num_classes=29):  # Ensure num_classes is set to 29 for the ASL dataset
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Ensure correct number of classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Load dataset again to get number of classes (make sure it's 29, not 2)
dataset_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\ASL_Dataset\asl_alphabet_train\asl_alphabet_train'
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_data = datasets.ImageFolder(dataset_path, transform=transform)
num_classes = len(train_data.classes)
print(f"Detected {num_classes} ASL classes.")  # Should print 29 if the dataset has 29 classes

# Load the trained model
model = ASLModel(num_classes)  # Ensure the number of classes is 29, matching your saved model
model.load_state_dict(torch.load(r'C:\Users\potte\asl interpreter\CV-Hackathon\asl_model_state_dict.pt'))
model.eval()

# Dummy input for the model (batch size 1, 3 channels, 64x64 image)
dummy_input = torch.randn(1, 3, 64, 64)

# Export the model to ONNX format
onnx_file_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\asl_model.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # Variable batch size for input
        "output": {0: "batch_size"}  # Variable batch size for output
    },
    opset_version=12  # Set the ONNX opset version (ensure compatibility)
)

print(f"Model has been converted to ONNX format and saved at {onnx_file_path}")
