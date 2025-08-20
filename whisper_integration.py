import whisper
import torch
import os
import json
import logging
from typing import Optional, Dict, Any, List

# Device & Model Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZE = "base"   # use "small" if you want better accuracy, but avoid medium/large on free tier

# Load Whisper model once globally
whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="whisper_integration.log")

def generate_timestamps(
    audio_path: str,
    output_json_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(audio_path):
            logging.error(f"[❌ ERROR] File not found: {audio_path}")
            return None

        logging.info(f"[📢 INFO] Transcribing audio: {audio_path}")
        result = whisper_model.transcribe(audio_path)

        text = result.get("text", "")
        segments = result.get("segments", [])

        if not segments:
            logging.error("[❌ ERROR] No segments returned from Whisper.")
            return None

        word_timestamps: List[Dict[str, Any]] = []

        # Approximate word-level timestamps by splitting segment duration
        for seg in segments:
            seg_text = seg["text"].strip()
            if not seg_text:
                continue

            words = seg_text.split()
            start, end = seg["start"], seg["end"]
            duration = end - start
            step = duration / max(len(words), 1)

            for i, word in enumerate(words):
                w_start = start + i * step
                w_end = w_start + step
                word_timestamps.append({
                    "word": word,
                    "start": round(w_start, 2),
                    "end": round(w_end, 2)
                })

        output = {
            "text": text,
            "timestamps": word_timestamps
        }

        logging.info(f"[✅ SUCCESS] Generated {len(word_timestamps)} word-level timestamps.")

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            logging.info(f"[💾 Saved JSON] {output_json_path}")

        return output

    except Exception as e:
        logging.error(f"[❌ ERROR] Whisper timestamp generation failed: {e}")
        return None
