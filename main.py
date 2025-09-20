import google.generativeai as genai
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import tempfile
import whisper
from dotenv import load_dotenv  


load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Use the API key from environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # <-- Fix here
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Load Whisper model once
whisper_model = whisper.load_model("base")

# -------------------------
# Helper Functions
# -------------------------

# --- Text Plagiarism Check ---
def check_text_plagiarism(text, reference_texts):
    prompt = f"""
    You are an AI plagiarism detector.
    Compare the following text against the reference texts for plagiarism.
    Text: "{text}"
    References: {reference_texts}
    Give a plagiarism score from 0 (no similarity) to 1 (identical), and briefly explain.
    """
    response = model.generate_content(prompt).text
    return response

# --- Audio to Text + Plagiarism ---
def audio_to_text(file_path):
    result = whisper_model.transcribe(file_path)
    transcript = result["text"]
    return transcript

# --- Video to Frames/Text + Plagiarism ---
def video_to_text(video_path):
    clip = VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".mp3")
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    # Transcribe audio via Whisper
    transcript = audio_to_text(audio_path)
    return transcript

# -------------------------
# Routes
# -------------------------

# Text input plagiarism
@app.route("/check/text", methods=["POST"])
def check_text():
    data = request.json
    input_text = data.get("text")
    references = data.get("references", [])
    result = check_text_plagiarism(input_text, references)
    return jsonify({"result": result})

# Audio input plagiarism
@app.route("/check/audio", methods=["POST"])
def check_audio():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    transcript = audio_to_text(file_path)
    references = request.form.getlist("references")
    result = check_text_plagiarism(transcript, references)
    return jsonify({"transcript": transcript, "result": result})

# Video input plagiarism
@app.route("/check/video", methods=["POST"])
def check_video():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    transcript = video_to_text(file_path)
    references = request.form.getlist("references")
    result = check_text_plagiarism(transcript, references)
    return jsonify({"transcript": transcript, "result": result})

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
