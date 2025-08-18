from flask import (
    Flask, request, render_template, redirect, url_for,
    flash, jsonify
)
from flask_bcrypt import Bcrypt
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.utils import secure_filename
from config import Config
from supabase_helper import upload_to_supabase
from supabase_db import (
    insert_story, insert_slide, insert_timestamps,
    update_slide_text_and_audio, delete_timestamps_by_slide,
    insert_dictionary_entry, get_slide_texts_by_story, get_user_story_titles,
    get_user_by_username, insert_user, update_story_title, get_story_owner,
    get_slides_by_story, get_slide_by_id, get_story_with_slides
)

from whisper_integration import generate_timestamps
import os
import json
import uuid
import logging
from supabase import create_client
from tts_engine import generate_tts


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure dictionary image folder exists
DICT_IMAGE_FOLDER = os.path.join("static", "Images", "dictionary")
os.makedirs(DICT_IMAGE_FOLDER, exist_ok=True)

app.secret_key = getattr(Config, 'SUPER_SECRET_KEY', 'secret')

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)


def process_tts_and_timestamps(slide_id, text):
    local_audio_path = generate_tts(text)
    if isinstance(local_audio_path, tuple):
        local_audio_path = local_audio_path[0]

    audio_url = None
    if local_audio_path and os.path.exists(local_audio_path):
        with open(local_audio_path, "rb") as f:
            audio_url = upload_to_supabase("audio", f, original_filename=os.path.basename(local_audio_path))

        delete_timestamps_by_slide(slide_id)

        timestamps = generate_timestamps(local_audio_path)
        if timestamps and "timestamps" in timestamps:
            insert_timestamps(slide_id, timestamps["timestamps"])

        os.remove(local_audio_path)

    if not audio_url:
        flash("Audio generation failed.", "danger")
    return audio_url


class User(UserMixin):
    def __init__(self, id_, username):
        self.id = id_
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    try:
        user = supabase.table("users").select("id, username").eq("id", user_id).single().execute().data
        if user:
            return User(user["id"], user["username"])
    except Exception:
        pass
    return None


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            insert_user(username, password_hash)
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Username already exists or error occurred.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = get_user_by_username(username)

        if user and bcrypt.check_password_hash(user['password_hash'], password):
            user_obj = User(user['id'], user['username'])
            login_user(user_obj)
            return jsonify({"success": True, "message": "Logged in successfully."})
        else:
            return jsonify({"success": False, "message": "Invalid credentials."}), 400

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')


@app.route('/upload_story', methods=['POST'])
@login_required
def upload_story():
    try:
        title = request.form.get('title')
        slide_count = int(request.form.get('slide_count', 0))

        if not title or slide_count <= 0:
            return "Missing title or invalid slide count", 400

        story_id = insert_story(title, user_id=current_user.id)

        for i in range(slide_count):
            text = request.form.get(f'slide_text_{i}', "")
            image_file = request.files.get(f'slide_image_{i}')
            audio_file = request.files.get(f'slide_audio_{i}')
            video_file = request.files.get(f'slide_video_{i}')
            json_file = request.files.get(f'slide_json_{i}')

            image_url = audio_url = video_url = None
            local_audio_path = None

            if image_file and image_file.filename:
                image_url = upload_to_supabase("images", image_file, original_filename=image_file.filename)

            if audio_file and audio_file.filename:
                audio_url = upload_to_supabase("audio", audio_file, original_filename=audio_file.filename)
                filename = secure_filename(audio_file.filename)
                local_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(local_audio_path)
            elif text.strip():
                local_audio_path = generate_tts(text)
                if isinstance(local_audio_path, tuple):
                    local_audio_path = local_audio_path[0]
                if local_audio_path and os.path.exists(local_audio_path):
                    with open(local_audio_path, "rb") as f:
                        audio_url = upload_to_supabase("audio", f, original_filename=os.path.basename(local_audio_path))

            if video_file and video_file.filename:
                video_url = upload_to_supabase("videos", video_file, original_filename=video_file.filename)

            slide_id = insert_slide(story_id, text, image_url, audio_url, video_url)

            timestamps = []
            if json_file and json_file.filename:
                try:
                    json_data = json.load(json_file)
                    word_data = json_data.get("words", json_data)
                    timestamps = word_data
                except Exception:
                    pass
            elif local_audio_path and os.path.exists(local_audio_path):
                whisper_result = generate_timestamps(local_audio_path)
                if whisper_result and "timestamps" in whisper_result:
                    timestamps = whisper_result["timestamps"]

            if timestamps:
                insert_timestamps(slide_id, timestamps)

            if local_audio_path and os.path.exists(local_audio_path):
                os.remove(local_audio_path)

        return redirect(url_for('upload'))

    except Exception as e:
        return f"An error occurred: {str(e)}", 500


@app.route('/modification', methods=['GET'])
@login_required
def modification():
    try:
        stories_data = get_user_story_titles(current_user.id)
        stories = [{"story_id": row["story_id"], "title": row["title"]} for row in stories_data]
        return render_template('modification.html', stories=stories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/modify_story/<int:story_id>', methods=['GET', 'POST'])
@login_required
def modify_story(story_id):
    try:
        print(f"📌 Trying to load story {story_id} for user {current_user.id}")
        story_title, slides = get_story_with_slides(story_id, current_user.id)

        if not story_title:
            print("❌ Story title not found or unauthorized access")
            flash('You are not authorized to modify this story.', 'danger')
            return redirect(url_for('modification'))

        print(f"✅ Loaded story '{story_title}' with {len(slides)} slides.")

        if request.method == 'POST':
            new_title = request.form.get('title')
            slide_ids = request.form.getlist('slide_ids[]')

            update_story_title(story_id, new_title)

            for slide_id in slide_ids:
                new_text = request.form.get(f"slide_text_{slide_id}")
                if new_text and new_text.strip():
                    audio_url = process_tts_and_timestamps(slide_id, new_text)
                    if audio_url:
                        update_slide_text_and_audio(slide_id, new_text, audio_url)

            flash("Story modified successfully.", "success")
            return redirect(url_for('modification'))

        return render_template(
    "modify_story.html",
    story={"story_id": story_id, "title": story_title},
    slides=slides,
    story_id=story_id  # 🔥 THIS LINE FIXES THE ERROR
)


    except Exception as e:
        print(f"🔥 Exception occurred in modify_story: {e}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('modification'))



@app.route("/dictionary")
@login_required
def dictionary():
    stories = get_user_story_titles(current_user.id)
    return render_template("dictionary.html", stories=stories)


@app.route("/dict_edit/<int:story_id>")
@login_required
def modify_story_dictionary(story_id):
    story = get_story_owner(story_id)
    if story != current_user.id:
        flash("Unauthorized access.", "danger")
        return redirect(url_for("dictionary"))

    slides = get_slide_texts_by_story(story_id)
    story_title = get_story_with_slides(story_id, current_user.id)[0]
    return render_template("dict_edit.html", story_id=story_id, story_title=story_title, slides=slides)


@app.route("/submit_dictionary_entries", methods=["POST"])
@login_required
def submit_dictionary_entries():
    user_id = current_user.id

    try:
        story_id = int(request.form.get("story_id"))
    except (ValueError, TypeError):
        flash("❌ Invalid story ID.", "danger")
        return redirect(url_for("dictionary"))

    words = request.form.getlist("word[]")
    images = request.files.getlist("image[]")

    if not words or not images:
        flash("❌ No dictionary words or images provided.", "danger")
        return redirect(url_for("dictionary"))

    if len(words) != len(images):
        flash("❌ Mismatch between number of words and images.", "danger")
        return redirect(url_for("dictionary"))

    for word, image in zip(words, images):
        word = word.strip()

        if not word or not image or image.filename == "":
            continue

        try:
            relative_path = upload_to_supabase("dictionary", image, image.filename)
            if not relative_path:
                flash(f"⚠️ Failed to upload image for word: {word}", "warning")
                continue

            insert_dictionary_entry(user_id, story_id, word, relative_path)

        except Exception as e:
            logging.error(f"Error saving entry '{word}': {e}")
            flash(f"❌ Could not save word: {word}", "danger")

    flash("✅ Dictionary entries uploaded successfully!", "success")
    return redirect(url_for("dictionary"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8080)
