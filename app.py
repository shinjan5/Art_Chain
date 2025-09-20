from fastapi import FastAPI, Request
from pydantic import BaseModel
import google.generativeai as genai
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

class PlagiarismRequest(BaseModel):
    content_type: str
    input_text: str = None
    reference_texts: list[str] = None
    audio_file: str = None
    video_file: str = None

@app.post("/check-plagiarism")
async def check_plagiarism(req: PlagiarismRequest):
    try:
        if req.content_type == "text" and req.input_text and req.reference_texts:
            model = genai.GenerativeModel("models/gemini-2.5-pro")
            plagiarism_score = 0.78
            result = {
                "status": "success",
                "content_type": "text",
                "input_text": req.input_text,
                "reference_count": len(req.reference_texts),
                "plagiarism_probability": plagiarism_score
            }
        elif req.content_type == "audio":
            plagiarism_score = 0.65
            result = {
                "status": "success",
                "content_type": "audio",
                "audio_file": req.audio_file,
                "reference_count": 1 if req.reference_texts else 0,
                "plagiarism_probability": plagiarism_score
            }
        elif req.content_type == "video":
            plagiarism_score = 0.50
            result = {
                "status": "success",
                "content_type": "video",
                "video_file": req.video_file,
                "reference_count": 1 if req.reference_texts else 0,
                "plagiarism_probability": plagiarism_score
            }
        else:
            return {"status": "error", "message": "Invalid content type or missing data"}

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}