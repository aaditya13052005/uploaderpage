import logging
from supabase import create_client
from config import Config

# Supabase setup
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# Logging
logging.basicConfig(level=logging.INFO, filename="supabase_errors.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------- Helper -------------------------
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

# ------------------------- Insert Functions -------------------------
def insert_story(title, user_id):
    if not title.strip():
        raise ValueError("Story title cannot be empty.")
    data = {"title": title, "user_id": user_id}
    result = supabase.table("stories").insert(data).execute()
    return result.data[0]["story_id"]

def insert_slide(story_id, text, image_path, audio_path, video_path):
    if not text.strip():
        raise ValueError("Slide text cannot be empty.")
    data = {
        "story_id": story_id,
        "text": text,
        "image_path": image_path or '',
        "audio_path": audio_path or '',
        "video_path": video_path or ''
    }
    result = supabase.table("slides").insert(data).execute()
    return result.data[0]["slide_id"]

def insert_timestamps(slide_id, timestamps):
    valid_entries = []
    for word_data in timestamps:
        word = str(word_data.get("word", "")).strip()
        start = safe_float(word_data.get("start"))
        end = safe_float(word_data.get("end"))
        if word and word.lower() != "word":
            valid_entries.append({
                "slide_id": slide_id,
                "word": word,
                "start_time": start,
                "end_time": end
            })
    if valid_entries:
        supabase.table("timestamps").insert(valid_entries).execute()

def insert_user(username, password_hash):
    supabase.table("users").insert({
        "username": username,
        "password_hash": password_hash
    }).execute()

def insert_dictionary_entry(user_id, story_id, word, image_path):
    if not word.strip():
        raise ValueError("Dictionary word cannot be empty.")
    supabase.table("word_dictionary").insert({
        "user_id": user_id,
        "story_id": story_id,
        "word": word,
        "image_path": image_path
    }).execute()

# ------------------------- Update Functions -------------------------
def update_slide_text_and_audio(slide_id, new_text, new_audio_url):
    supabase.table("slides").update({
        "text": new_text,
        "audio_path": new_audio_url
    }).eq("slide_id", slide_id).execute()

def update_story_title(story_id, new_title):
    supabase.table("stories").update({
        "title": new_title
    }).eq("story_id", story_id).execute()

def update_slide_with_timestamps(slide_id, new_text, new_audio_url, new_timestamps):
    update_slide_text_and_audio(slide_id, new_text, new_audio_url)
    delete_timestamps_by_slide(slide_id)
    insert_timestamps(slide_id, new_timestamps)

# ------------------------- Delete Functions -------------------------
def delete_timestamps_by_slide(slide_id):
    supabase.table("timestamps").delete().eq("slide_id", slide_id).execute()

# ------------------------- Get / Fetch Functions -------------------------
def get_user_by_username(username):
    result = supabase.table("users").select("id, username, password_hash").eq("username", username).single().execute()
    return result.data

def get_user_stories(user_id):
    result = supabase.table("stories").select("*").eq("user_id", user_id).execute()
    return result.data

def get_slides_by_story(story_id):
    result = supabase.table("slides").select("*").eq("story_id", story_id).execute()
    return result.data

def get_story_with_slides(story_id, user_id):
    try:
        story_resp = supabase.table("stories").select("title").eq("story_id", story_id).eq("user_id", user_id).single().execute()
        story = story_resp.data
        if not story:
            return None, []

        title = story["title"]

        slides_resp = supabase.table("slides").select("slide_id, text, story_id").eq("story_id", story_id).order("slide_id", desc=False).execute()
        slides = slides_resp.data or []

        return title, slides

    except Exception as e:
        logging.error(f"Error in get_story_with_slides: {e}")
        return None, []


def get_slide_by_id(slide_id):
    result = supabase.table("slides").select("*").eq("slide_id", slide_id).single().execute()
    return result.data

def get_story_owner(story_id):
    result = supabase.table("stories").select("user_id").eq("story_id", story_id).single().execute().data
    return result["user_id"] if result else None

def get_timestamps_by_slide(slide_id):
    result = supabase.table("timestamps").select("word, start_time, end_time").eq("slide_id", slide_id).execute()
    return result.data

def get_user_story_titles(user_id):
    result = supabase.table("stories").select("story_id, title").eq("user_id", user_id).execute()
    return result.data

def get_slide_texts_by_story(story_id):
    result = supabase.table("slides").select("slide_id, text").eq("story_id", story_id).execute()
    return result.data
