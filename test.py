import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the ASLModel class (matching model.py)
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

# Initialize the model with correct class count
num_classes = 29  # A-Z + space, delete, nothing
model = ASLModel(num_classes)

# Load the state_dict (weights) into the model
model_path = r'/workspaces/CV-Hackathon/asl_model_state_dict.pt'
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Ensure compatibility

# Switch to evaluation mode
model.eval()

# Define the same transformations as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the image
img_path = r'/workspaces/CV-Hackathon/ASL_Dataset/asl_alphabet_test/asl_alphabet_test/P_test.jpg'
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Check image shape
print(f"Image shape: {image.shape}")  # Should be (1, 3, 64, 64)

# Run inference
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

# Define ASL classes
asl_classes = [chr(i) for i in range(65, 91)] + ['space', 'delete', 'nothing']

print(f"Predicted Letter: {asl_classes[predicted_class]}")
