import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Define the same transformations as training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the image
img_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\ASL_Dataset\asl_alphabet_test\asl_alphabet_test\Z_test.jpg'
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Check image shape
print(f"Image shape: {image.shape}")  # Should be (1, 3, 64, 64)

# Convert image to numpy array
image_np = image.numpy()

# Load the ONNX model
onnx_model_path = r'C:\Users\potte\asl interpreter\CV-Hackathon\static\asl_model.onnx'
session = ort.InferenceSession(onnx_model_path)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: image_np})

# Get the predicted class
output = result[0]
predicted_class = np.argmax(output, axis=1).item()

# Define ASL classes
asl_classes = [chr(i) for i in range(65, 91)] + ['space', 'delete', 'nothing']

print(f"Predicted Letter: {asl_classes[predicted_class]}")
