# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import JSONResponse
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import io

# router = APIRouter(
#     tags=["Prediction"]
# )

# # Load mô hình một lần duy nhất
# model = tf.keras.models.load_model("plant_disease_model.keras")

# # Lấy class_names từ train_generator (giả định đã lưu ra file hoặc hardcode)
# CLASS_NAMES = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#     'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
#     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
#     'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#     'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
#     'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
#     'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
#     'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
#     'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# # Tiền xử lý ảnh (giống notebook logic)
# def preprocess(img_bytes: bytes) -> np.ndarray:
#     img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # API upload ảnh → dự đoán
# @router.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         img_bytes = await file.read()
#         img_array = preprocess(img_bytes)

#         pred = model.predict(img_array)
#         pred_index = np.argmax(pred)
#         pred_label = CLASS_NAMES[pred_index]
#         confidence = float(np.max(pred))

#         return {
#             "predicted_class": pred_label,
#             "confidence": round(confidence, 4)
#         }
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import io

router = APIRouter(tags=["Prediction"])

# --- Load model từ Hugging Face ---
model_path = hf_hub_download(
    repo_id="yenly1234/Plantvillage",
    filename="plant_disease_model.keras"
)
model = tf.keras.models.load_model(model_path)

# Danh sách class gốc từ thư mục
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

# Tạo label dễ đọc
CLASS_INDEX_TO_LABEL = {}
for idx, label in enumerate(CLASS_NAMES):
    if "___" in label:
        plant, disease = label.split("___")
        disease = disease.replace("_", " ")
        if disease.lower() == "healthy":
            display_label = f"{plant} healthy"
        elif plant.lower() in disease.lower():
            display_label = disease
        else:
            display_label = f"{plant} {disease}"
    else:
        display_label = label.replace("_", " ")
    CLASS_INDEX_TO_LABEL[idx] = display_label

# Tiền xử lý ảnh
def preprocess(img_bytes: bytes) -> np.ndarray:
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# API dự đoán
@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_array = preprocess(img_bytes)

        pred = model.predict(img_array)
        pred_index = int(np.argmax(pred))
        confidence = float(np.max(pred))
        pred_label = CLASS_INDEX_TO_LABEL[pred_index]

        return {
            "class_index": pred_index,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
