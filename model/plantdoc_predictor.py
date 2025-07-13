import torch
import numpy as np
import cv2
import pathlib
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision import transforms

# ✅ sửa đúng đường dẫn import
from model.model import build_model
from model.class_names import class_names as CLASS_NAMES


# Tree dictionary
TREE_DICT = {
    "Apple": ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf"],
    "Bell_pepper": ["Bell_pepper leaf", "Bell_pepper leaf spot"],
    "Blueberry": ["Blueberry leaf"],
    "Cherry": ["Cherry leaf"],
    "Corn": ["Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf"],
    "grape": ["grape leaf", "grape leaf black rot"],
    "Peach": ["Peach leaf"],
    "Potato": ["Potato leaf early blight", "Potato leaf late blight"],
    "Raspberry": ["Raspberry leaf"],
    "Soyabean": ["Soyabean leaf"],
    "Squash": ["Squash Powdery mildew leaf"],
    "Strawberry": ["Strawberry leaf"],
    "Tomato": [
        "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf",
        "Tomato leaf bacterial spot", "Tomato leaf late blight",
        "Tomato leaf mosaic virus", "Tomato leaf yellow virus", "Tomato mold leaf"
    ]
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_RESIZE = 224

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = None

def load_model():
    global model
    if model is None:
        path = hf_hub_download(
        repo_id="yenly1234/plandocs",
        filename="best_model.pth"
        )
        checkpoint = torch.load(path, map_location=DEVICE)
        model_name = "efficientnetb0"  # Nếu bạn đặt tên rõ hơn thì đọc từ path cũng được
        model = build_model(
            model_name=model_name,
            fine_tune=False,
            num_classes=checkpoint['model_state_dict']['classifier.1.weight'].shape[0]
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model

def predict_plantdoc(img_bytes):
    image_np = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    model = load_model()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    top_idx = np.argmax(probs)
    top_class = CLASS_NAMES[top_idx]
    plant_name = top_class.split()[0]

    related = TREE_DICT.get(plant_name, [])
    results = []

    for disease in related:
        if disease in CLASS_NAMES:
            idx = CLASS_NAMES.index(disease)
            results.append({"disease": disease, "confidence": round(float(probs[idx]), 4)})

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    return {
        "plant": plant_name,
        "top_prediction": top_class,
        "disease_confidences": results
    }
