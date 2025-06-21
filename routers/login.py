from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(data: LoginRequest):
    db_path = os.path.join("db_data", "database", "my_database.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, full_name, username FROM login WHERE username = ? AND password = ?",
        (data.username, data.password)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        return {
            "message": "✅ Đăng nhập thành công",
            "user": {
                "id": user[0],
                "full_name": user[1],
                "username": user[2]
            }
        }
    else:
        raise HTTPException(status_code=401, detail="❌ Sai username hoặc password")
