from fastapi import FastAPI
from routers import login  # import file login.py từ thư mục routers

app = FastAPI()

# Đăng ký router từ file login.py
app.include_router(login.router)
