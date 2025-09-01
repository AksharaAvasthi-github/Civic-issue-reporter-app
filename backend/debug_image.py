import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os

from torchvision.models import resnet18, ResNet18_Weights

def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_urgency_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict urgency
def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    class_names = ['low', 'high']
    return class_names[predicted.item()]

if __name__ == "__main__":
    # Get image path from user input
    image_path = input("Paste the full image path: ").strip('"')

    if not os.path.exists(image_path):
        print("❌ File not found. Check the path and try again.")
    else:
        model = load_model()
        prediction = predict(image_path, model)
        print(f"✅ Predicted Urgency: {prediction.upper()}")
