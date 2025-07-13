import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
import io

# --- Load model từ HuggingFace ---
model_path = hf_hub_download(
    repo_id="yenly1234/Plantvillage",
    filename="plant_disease_model.keras"
)
model = tf.keras.models.load_model(model_path)

# --- Class gốc từ model ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- Xử lý nhãn dễ đọc ---
CLASS_INDEX_TO_LABEL = {}
for idx, label in enumerate(CLASS_NAMES):
    if "___" in label:
        plant, disease = label.split("___")
        disease = disease.replace("_", " ")
        plant = plant.replace("(", "").replace(")", "").replace(",", "").replace("_", " ")
        if disease.lower() == "healthy":
            display_label = f"{plant} healthy"
        else:
            display_label = f"{plant} {disease}"
    else:
        display_label = label.replace("_", " ")
    CLASS_INDEX_TO_LABEL[idx] = display_label

# --- Dictionary nhóm bệnh theo cây ---
TREE_DICT = {}
for idx, label in CLASS_INDEX_TO_LABEL.items():
    plant = label.split()[0]
    TREE_DICT.setdefault(plant, []).append(label)

# --- Preprocess ảnh ---
def preprocess(img_bytes: bytes) -> np.ndarray:
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Hàm predict chính ---
def predict_plantvillage(img_bytes: bytes):
    img_array = preprocess(img_bytes)
    preds = model.predict(img_array)[0]  # shape (num_classes,)

    top_idx = int(np.argmax(preds))
    top_label = CLASS_INDEX_TO_LABEL[top_idx]
    plant_name = top_label.split()[0]

    related_labels = TREE_DICT.get(plant_name, [])
    results = []

    for label in related_labels:
        class_idx = next((i for i, v in CLASS_INDEX_TO_LABEL.items() if v == label), None)
        if class_idx is not None:
            results.append({
                "disease": label,
                "confidence": round(float(preds[class_idx]), 4)
            })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return {
        "plant": plant_name,
        "top_prediction": top_label,
        "disease_confidences": results
    }
