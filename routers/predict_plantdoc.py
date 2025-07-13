from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from model.plantdoc_predictor import predict_plantdoc

router = APIRouter()

@router.post("/predict/plantdoc")
async def predict_plantdoc_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_plantdoc(contents)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

