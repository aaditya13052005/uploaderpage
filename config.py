import os

class Config:
    # Supabase Credentials
    SUPABASE_URL = "https://fxwuqyiecicagoqdckjn.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4d3VxeWllY2ljYWdvcWRja2puIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM0NzU1MDQsImV4cCI6MjA1OTA1MTUwNH0.7LgSNvAwKV2ORDGp8bAj5Nln3krN-jbbmbLFLS7GKvA"
    SUPABASE_BUCKET = "stories"

    # MySQL Credentials
    MYSQL_HOST = "localhost"
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "1310"  # ⬅ Replace with your actual password
    MYSQL_DB = "vidya_vahini"

    # Upload Settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'mp4', 'json'}

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
