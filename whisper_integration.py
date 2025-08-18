import whisper
import whisperx
import torch
import os
import json
import logging
from typing import Optional, Dict, Any, List

# Device & Model Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MODEL_SIZE = "medium"

# Load Whisper model once globally
whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="whisper_integration.log")

def get_align_model(language: str):
    """
    Loads the alignment model and metadata for the specified language.
    """
    model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
    return model_a, metadata

def generate_timestamps(
    audio_path: str,
    output_json_path: Optional[str] = None,
    method: str = "default"  # Ignored for now, retained for forward compatibility
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

        logging.info("[📢 INFO] Performing alignment with WhisperX...")
        model_a, metadata = get_align_model(language=result["language"])

        aligned_result = whisperx.align(
            segments,
            model_a,
            metadata,
            audio_path,
            DEVICE,
        )

        word_segments = aligned_result.get("word_segments", [])
        if not word_segments:
            logging.error("[❌ ERROR] No word segments found in alignment.")
            return None

        word_timestamps: List[Dict[str, Any]] = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for w in word_segments
        ]

        output = {
            "text": text,
            "timestamps": word_timestamps
        }

        logging.info(f"[✅ SUCCESS] Generated {len(word_timestamps)} timestamps.")

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            logging.info(f"[💾 Saved JSON] {output_json_path}")

        return output

    except Exception as e:
        logging.error(f"[❌ ERROR] Whisper timestamp generation failed: {e}")
        return None
