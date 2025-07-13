from fastapi import FastAPI
from routers import login  # import file login.py từ thư mục routers
from routers import predict 
# from routers import predict_plantdoc
app = FastAPI()

# Đăng ký router từ file login.py
app.include_router(login.router)
app.include_router(predict.router)
# app.include_router(predict_plantdoc.router)