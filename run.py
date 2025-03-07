from app import create_app, precheck
from dotenv import load_dotenv
import os

load_dotenv()

app = precheck()

if app:  # Pastikan app tidak None sebelum dijalankan
    if __name__ == '__main__':
        app.run()
else:
    print("Application failed to start due to missing folders.")
