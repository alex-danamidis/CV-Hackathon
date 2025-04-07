import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the ASLModel class (same as in your training script)
class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*14*14, 128)  # Ensure this matches your architecture
        self.fc2 = nn.Linear(128, 29)  # Adjusted for 29 classes (A-Z + space, delete, nothing)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = ASLModel()

# Load the state_dict (weights) into the model
model_path = r'C:\Users\Tyler\Desktop\Programming\Projects\alex-hackathon\CV-Hackathon\asl_model_state_dict.pt'
model.load_state_dict(torch.load(model_path))  # Load only the state dict (weights)

# Switch the model to evaluation mode
model.eval()

# Define the same transformations as during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training normalization
])

# Test with a sample image
img_path = r'C:\Users\Tyler\Desktop\Programming\Projects\alex-hackathon\CV-Hackathon\ASL_Dataset\asl_alphabet_test\asl_alphabet_test\Y_test.jpg'  # Example path
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Check image shape
print(f"Image shape: {image.shape}")  # It should be (1, 3, 64, 64)

# Run inference on the image
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

# Define ASL classes (ensure this matches your training)
asl_classes = [chr(i) for i in range(65, 91)] + ['space', 'delete', 'nothing']

print(f"Predicted Letter: {asl_classes[predicted_class]}")
