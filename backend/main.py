import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from backend.utils import (
    analyze_text_sentiment,
    analyze_image_heuristics,
    analyze_video_heuristics,
    call_gemini_api
)

app = FastAPI(title="Multimodal Sentiment Analysis Backend API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

class TweetRequest(BaseModel):
    username: str
    count: Optional[int] = 5
    use_live: Optional[bool] = False

# Ensure temp directory exists inside workspace
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"status": "running", "message": "Multimodal Sentiment Analysis API is active"}

@app.post("/analyze/text")
def analyze_text(payload: TextPayload):
    try:
        result = analyze_text_sentiment(payload.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    gemini_api_key: Optional[str] = Form(None)
):
    # Check valid file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")
        
    # Save file temporarily
    temp_file_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Run local heuristics
        local_result = analyze_image_heuristics(temp_file_path)
        
        # 2. Run Gemini if key is provided
        if gemini_api_key and gemini_api_key.strip():
            try:
                # Read image bytes
                with open(temp_file_path, "rb") as f:
                    image_bytes = f.read()
                    
                mime_type = "image/png" if ext == ".png" else "image/jpeg"
                prompt = (
                    "Analyze the sentiment of this image. Categorize it strictly as 'Positive', 'Negative', "
                    "or 'Neutral'. Provide a confidence score (0.0 to 1.0) and a brief description of "
                    "the visual mood, emotional cues, objects, and facial expressions (if any). "
                    "Return ONLY a JSON object with the keys: 'sentiment' (string), 'confidence' (float), "
                    "'mood_description' (string)."
                )
                
                gemini_result = call_gemini_api(
                    api_key=gemini_api_key,
                    prompt=prompt,
                    image_bytes_list=[image_bytes],
                    mime_types=[mime_type]
                )
                
                # Merge and prioritize Gemini results
                return {
                    "method": "Gemini Multimodal AI",
                    "sentiment": gemini_result.get("sentiment", local_result["sentiment"]),
                    "confidence": gemini_result.get("confidence", 0.90),
                    "mood_description": gemini_result.get("mood_description", local_result["mood_description"]),
                    "heuristics": local_result
                }
            except Exception as gem_err:
                print(f"Gemini API error: {gem_err}. Falling back to heuristics.")
                return {
                    "method": "Local Heuristics (Gemini Failed)",
                    "error": str(gem_err),
                    **local_result
                }
                
        return {
            "method": "Local Visual Heuristics",
            **local_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    gemini_api_key: Optional[str] = Form(None)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")
        
    temp_file_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Run local heuristics (motion, brightness timeline)
        local_result = analyze_video_heuristics(temp_file_path)
        local_result["sentiment"] = local_result["overall_sentiment"]
        
        # 2. Run Gemini if key is provided
        if gemini_api_key and gemini_api_key.strip():
            try:
                # Sample key frames for Gemini
                import cv2
                cap = cv2.VideoCapture(temp_file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Sample 5 frames spaced evenly
                indices = [int(i * (total_frames - 1) / 4) for i in range(5)] if total_frames > 5 else list(range(total_frames))
                
                image_bytes_list = []
                mime_types = []
                
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in indices:
                        # Encode frame as JPEG
                        success, encoded_image = cv2.imencode('.jpg', frame)
                        if success:
                            image_bytes_list.append(encoded_image.tobytes())
                            mime_types.append("image/jpeg")
                    frame_idx += 1
                cap.release()
                
                if image_bytes_list:
                    prompt = (
                        "The attached images are sequential frames from a video or Instagram Reel. "
                        "Analyze this sequence to determine the video's overall sentiment ('Positive', 'Negative', 'Neutral'). "
                        "Identify the pacing, colors, scene context, and emotion. "
                        "Provide: 1. overall_sentiment (string: Positive, Negative, Neutral) "
                        "2. confidence (float: 0.0 to 1.0) "
                        "3. explanation (string summary) "
                        "4. timeline (array of objects with keys 'frame_index' (int), 'sentiment' (string), and 'description' (string) for each frame). "
                        "Return ONLY a JSON object containing these 4 keys."
                    )
                    
                    gemini_result = call_gemini_api(
                        api_key=gemini_api_key,
                        prompt=prompt,
                        image_bytes_list=image_bytes_list,
                        mime_types=mime_types
                    )
                    
                    # Merge local timeline with Gemini analysis
                    return {
                        "method": "Gemini Multimodal AI",
                        "sentiment": gemini_result.get("overall_sentiment", local_result["overall_sentiment"]),
                        "confidence": gemini_result.get("confidence", 0.95),
                        "explanation": gemini_result.get("explanation", "Successful multimodal analysis."),
                        "gemini_timeline": gemini_result.get("timeline", []),
                        "heuristics": local_result
                    }
            except Exception as gem_err:
                print(f"Gemini API video error: {gem_err}. Falling back to heuristics.")
                return {
                    "method": "Local Heuristics (Gemini Failed)",
                    "error": str(gem_err),
                    **local_result
                }
                
        return {
            "method": "Local Video Heuristics",
            **local_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/analyze/tweets")
def analyze_tweets(payload: TweetRequest):
    username = payload.username.strip().replace("@", "")
    tweets = []
    
    # 1. Optional Live scraping with Nitter (experimental)
    if payload.use_live:
        try:
            from ntscraper import Nitter
            scraper = Nitter(log_level=1)
            tweets_data = scraper.get_tweets(username, mode='user', number=payload.count)
            if 'tweets' in tweets_data and tweets_data['tweets']:
                for t in tweets_data['tweets']:
                    tweet_text = t.get('text', '')
                    if tweet_text:
                        analysis = analyze_text_sentiment(tweet_text)
                        tweets.append({
                            "id": t.get('link', ''),
                            "date": t.get('date', 'Unknown'),
                            "text": tweet_text,
                            "stats": {
                                "likes": t.get('stats', {}).get('likes', 0),
                                "retweets": t.get('stats', {}).get('retweets', 0)
                            },
                            **analysis
                        })
        except Exception as e:
            print(f"Live scraping failed: {e}. Falling back to Mock Tweets.")
            
    # 2. Mock Tweets Fallback (Guarantees elegant visuals and no API blockages)
    if not tweets:
        import random
        from datetime import datetime, timedelta
        
        # Seed random based on username length to keep it semi-consistent for the user
        random.seed(len(username))
        
        positive_templates = [
            "Just launched our new product dashboard! So excited to see everyone's feedback. 🚀✨ #tech #dashboard",
            "What a beautiful morning! Had a great coffee and ready to code all day. ☕💻 Feeling very productive!",
            "Huge thank you to the team for support on the latest project. We did it! 🎉🙌",
            "This sentiment analysis tool works incredibly well! Streamlit and FastAPI make a killer combo. ❤️",
            "Loving the feedback on Instagram reels recently, positive vibes only!"
        ]
        negative_templates = [
            "Spent 3 hours debugging a simple typos in config. So frustrating... 😠 #developerlife",
            "The service is down again. Really disappointed with the customer support this week. 📉",
            "Traffic is horrible today. Going to be late for the sprint meeting. 🚗🚗💨",
            "Not happy with the performance regression in the latest build. We need to optimize.",
            "Why is scraping tweets so difficult now? Everything is rate limited or blocked. 😭"
        ]
        neutral_templates = [
            "We are holding a team meeting tomorrow at 10 AM EST.",
            "Python 3.12 has some interesting updates. Reading the release notes now.",
            "Quick update: the server maintenance is scheduled for Sunday night.",
            "Comparing different visualization libraries for our next dashboard layout. Any suggestions?",
            "Just finished watching the new documentary on machine learning. Interesting perspective."
        ]
        
        for i in range(payload.count):
            sent_type = random.choice(["Positive", "Negative", "Neutral"])
            if sent_type == "Positive":
                text = random.choice(positive_templates)
            elif sent_type == "Negative":
                text = random.choice(negative_templates)
            else:
                text = random.choice(neutral_templates)
                
            analysis = analyze_text_sentiment(text)
            date_str = (datetime.now() - timedelta(days=i)).strftime("%b %d, %Y")
            
            tweets.append({
                "id": f"mock_tweet_{i}",
                "date": date_str,
                "text": text,
                "stats": {
                    "likes": random.randint(10, 500),
                    "retweets": random.randint(2, 80)
                },
                **analysis
            })
            
    # Calculate summary metrics
    sentiments = [t["sentiment"] for t in tweets]
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    neu_count = sentiments.count("Neutral")
    
    return {
        "username": username,
        "is_mock": not payload.use_live or len(tweets) > 0 and "mock" in tweets[0]["id"],
        "count": len(tweets),
        "metrics": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count,
            "positive_ratio": round(pos_count / len(tweets), 2) if tweets else 0
        },
        "tweets": tweets
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
