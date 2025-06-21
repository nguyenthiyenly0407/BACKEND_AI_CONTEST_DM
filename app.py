import sqlite3
import os

os.makedirs("database", exist_ok=True)


conn = sqlite3.connect("database/my_database.db")
cursor = conn.cursor()


cursor.execute('''
CREATE TABLE IF NOT EXISTS login (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')


users = [
    ("Nguyễn Thị Yến Ly", "yenly", "123456"),
    ("Phan Thị Quỳnh Như", "quynhnhu", "123456")
]

for user in users:
    try:
        cursor.execute('''
            INSERT INTO login (full_name, username, password)
            VALUES (?, ?, ?)
        ''', user)
    except sqlite3.IntegrityError:
        print(f"⚠️ Username '{user[1]}' đã tồn tại, bỏ qua insert.")
        
cursor.execute("SELECT * FROM login")
rows = cursor.fetchall()

print("\n📋 Dữ liệu trong bảng login:")
for row in rows:
    print(row)

conn.commit()
conn.close()


