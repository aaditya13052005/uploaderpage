import uuid
import os
import asyncio
import logging
from werkzeug.utils import secure_filename
from config import Config
import edge_tts

# Setup
OUTPUT_DIR = Config.UPLOAD_FOLDER
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="audio_generation.log")

# Default voice (you can customize this per language or user)
DEFAULT_VOICE = "en-US-JennyNeural"

async def edge_tts_generate(text, voice=DEFAULT_VOICE):
    try:
        file_name = secure_filename(f"{uuid.uuid4()}.mp3")
        output_path = os.path.join(OUTPUT_DIR, file_name)

        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

        logging.info(f"✅ Edge TTS audio saved: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"[ERROR] Edge TTS failed: {e}")
        return None

def generate_tts(text, voice=DEFAULT_VOICE):
    return asyncio.run(edge_tts_generate(text, voice=voice))
