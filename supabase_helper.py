import os
import uuid
import mimetypes
import logging
from werkzeug.utils import secure_filename
from supabase import create_client, Client
from PIL import Image
from pydub.utils import mediainfo
from config import Config

# ------------------------- Logging Setup -------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------- Supabase Client Init -------------------------
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# ------------------------- Detect File Extension -------------------------
def detect_file_extension(file_path):
    try:
        # Check image type
        try:
            with Image.open(file_path) as img:
                img.verify()
            logging.info(f"Detected image: {file_path}")
            return ".png"
        except Exception:
            pass

        # Check audio type
        try:
            info = mediainfo(file_path)
            if info.get("duration") and "audio" in info.get("codec_type", ""):
                logging.info(f"Detected audio: {file_path}")
                return ".mp3"
        except Exception:
            pass

        # Check video type
        try:
            info = mediainfo(file_path)
            if info.get("duration") and info.get("video"):
                logging.info(f"Detected video: {file_path}")
                return ".mp4"
        except Exception:
            pass

        logging.warning(f"Unknown file type for {file_path}, defaulting to .bin")
        return ".bin"

    except Exception as e:
        logging.error(f"Extension detection error: {e}")
        return ".bin"

# ------------------------- Upload File to Supabase -------------------------
def upload_to_supabase(subfolder, file, original_filename=None, delete_temp=True, overwrite=False):
    try:
        temp_dir = Config.UPLOAD_FOLDER
        os.makedirs(temp_dir, exist_ok=True)

        # Save file temporarily
        temp_placeholder = os.path.join(temp_dir, f"{uuid.uuid4()}")
        if hasattr(file, "save"):
            file.save(temp_placeholder)
        elif hasattr(file, "read"):
            with open(temp_placeholder, "wb") as out_file:
                out_file.write(file.read())
        elif isinstance(file, str) and os.path.exists(file):
            temp_placeholder = file
        else:
            raise ValueError("Unsupported file type")

        # Determine extension
        ext = os.path.splitext(original_filename or "")[1].lower()
        if not ext or ext not in ['.mp3', '.wav', '.png', '.jpg', '.jpeg', '.webp', '.mp4']:
            ext = detect_file_extension(temp_placeholder)
        if not ext.startswith("."):
            ext = "." + ext

        final_filename = secure_filename(f"{uuid.uuid4()}{ext}")
        final_path = os.path.join(temp_dir, final_filename)
        if temp_placeholder != final_path:
            os.rename(temp_placeholder, final_path)

        content_type, _ = mimetypes.guess_type(final_path)
        content_type = content_type or "application/octet-stream"
        bucket_path = f"{subfolder}/{final_filename}"

        # Check for existing file if overwrite is disabled
        if not overwrite:
            folder = os.path.dirname(bucket_path)
            filename = os.path.basename(bucket_path)
            existing = supabase.storage.from_(Config.SUPABASE_BUCKET).list(folder)
            if any(f["name"] == filename for f in existing):
                logging.warning(f"File already exists: {bucket_path}")
                return bucket_path

        # Upload
        with open(final_path, "rb") as f:
            res = supabase.storage.from_(Config.SUPABASE_BUCKET).upload(
                bucket_path, f, {"content-type": content_type}
            )

        # Cleanup
        if delete_temp and os.path.exists(final_path):
            os.remove(final_path)

        if getattr(res, "error", None):
            logging.error(f"Supabase upload failed: {res.error.message}")
            raise Exception(f"Upload failed: {res.error.message}")

        logging.info(f"Uploaded to Supabase: {bucket_path}")
        return bucket_path

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return None
