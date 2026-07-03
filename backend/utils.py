import os
import re
import cv2
import pickle
import numpy as np
from PIL import Image
import requests
import base64
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Global variables to cache model & vectorizer
_model = None
_vectorizer = None

def load_local_model():
    """Load local model and vectorizer once."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(workspace_dir, 'model.pkl')
        vectorizer_path = os.path.join(workspace_dir, 'vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                _model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                _vectorizer = pickle.load(f)
        else:
            print(f"Warning: Model or vectorizer not found at {model_path} or {vectorizer_path}")
    return _model, _vectorizer

def clean_text(text: str) -> str:
    """Clean and preprocess text for the TF-IDF model."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def analyze_text_sentiment(text: str) -> dict:
    """Analyze text sentiment using local model (if available) and NLTK VADER."""
    model, vectorizer = load_local_model()
    
    # NLTK VADER score
    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']
    
    if compound >= 0.05:
        vader_sentiment = "Positive"
    elif compound <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"
        
    # Local Model prediction (binary: Positive/Negative)
    model_sentiment = "Unknown"
    if model is not None and vectorizer is not None:
        try:
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            pred = model.predict(vectorized)
            model_sentiment = "Negative" if pred[0] == 0 else "Positive"
        except Exception as e:
            print(f"Error in model prediction: {e}")
            
    # Combine or choose sentiment (prefer VADER for multi-class Neutral)
    final_sentiment = vader_sentiment
    
    return {
        "text": text,
        "sentiment": final_sentiment,
        "score": compound,
        "vader_scores": vader_scores,
        "local_model_sentiment": model_sentiment
    }

def analyze_image_heuristics(image_path: str) -> dict:
    """Perform local visual heuristics analysis on an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file.")
        
    # Get image dimensions
    h, w, c = img.shape
    
    # Calculate brightness and contrast (on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Calculate average RGB values
    # OpenCV uses BGR, so convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_r = float(np.mean(img_rgb[:, :, 0]))
    mean_g = float(np.mean(img_rgb[:, :, 1]))
    mean_b = float(np.mean(img_rgb[:, :, 2]))
    
    # Dominant color heuristic (extract top colors by resizing to 4x4 grid)
    small_img = cv2.resize(img_rgb, (4, 4), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape(-1, 3)
    
    # Find unique colors or simply pick diverse ones
    unique_colors = []
    for p in pixels:
        color_hex = f"#{p[0]:02x}{p[1]:02x}{p[2]:02x}"
        if color_hex not in unique_colors:
            unique_colors.append(color_hex)
    dominant_colors = unique_colors[:5]
    
    # Calculate mood heuristic
    # Warmth: Higher Red/Green ratio, Coolness: Higher Blue ratio
    warmth = mean_r / (mean_b + 1e-5)
    
    if brightness > 140:
        if warmth > 1.1:
            sentiment = "Positive"
            mood = "Warm & Bright (Optimistic, Welcoming)"
        else:
            sentiment = "Positive"
            mood = "Cool & Bright (Clean, Calm, Peaceful)"
    elif brightness < 80:
        if contrast > 60:
            sentiment = "Negative"
            mood = "Dark & High Contrast (Dramatic, Intense, Somber)"
        else:
            sentiment = "Negative"
            mood = "Dark & Low Contrast (Gloomy, Mysterious, Cold)"
    else:
        sentiment = "Neutral"
        mood = "Balanced Light (Neutral, Natural)"
        
    return {
        "dimensions": f"{w}x{h}",
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "average_rgb": {"R": round(mean_r, 1), "G": round(mean_g, 1), "B": round(mean_b, 1)},
        "dominant_colors": dominant_colors,
        "mood_description": mood,
        "sentiment": sentiment,
        "confidence": 0.70
    }

def analyze_video_heuristics(video_path: str) -> dict:
    """Perform local visual & motion heuristics analysis on a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Target frame sampling (up to 30 frames total)
    sample_rate = max(1, total_frames // 30)
    
    timeline = []
    prev_gray = None
    frame_idx = 0
    sampled_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Gray for brightness/contrast & motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Compute motion activity (difference from previous sampled frame)
            motion = 0.0
            if prev_gray is not None:
                # Resize to make diff computation fast and scale-invariant
                g1 = cv2.resize(prev_gray, (100, 100))
                g2 = cv2.resize(gray, (100, 100))
                diff = cv2.absdiff(g1, g2)
                motion = float(np.mean(diff))
                
            prev_gray = gray.copy()
            
            # Average RGB values
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_r = float(np.mean(rgb[:, :, 0]))
            mean_g = float(np.mean(rgb[:, :, 1]))
            mean_b = float(np.mean(rgb[:, :, 2]))
            warmth = mean_r / (mean_b + 1e-5)
            
            # Sub-frame Sentiment Estimation
            if brightness > 130:
                frame_sent = "Positive"
            elif brightness < 75:
                frame_sent = "Negative"
            else:
                frame_sent = "Neutral"
                
            timeline.append({
                "timestamp": round(timestamp, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "motion": round(motion, 2),
                "r": round(mean_r, 1),
                "g": round(mean_g, 1),
                "b": round(mean_b, 1),
                "sentiment": frame_sent
            })
            sampled_count += 1
            
        frame_idx += 1
        
    cap.release()
    
    if not timeline:
        raise ValueError("No frames could be extracted from video.")
        
    # Aggregate sentiment counts
    sentiments = [t['sentiment'] for t in timeline]
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    neu_count = sentiments.count("Neutral")
    
    # Calculate overall sentiment
    if pos_count > neg_count and pos_count > neu_count:
        overall_sentiment = "Positive"
    elif neg_count > pos_count and neg_count > neu_count:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
        
    # Calculate dynamics metrics
    avg_motion = float(np.mean([t['motion'] for t in timeline]))
    avg_brightness = float(np.mean([t['brightness'] for t in timeline]))
    
    # Motion description
    if avg_motion > 15:
        pacing = "Fast-Paced (High Energy / Action)"
    elif avg_motion > 5:
        pacing = "Moderately Paced (Normal / Engaging)"
    else:
        pacing = "Slow-Paced (Still / Cinematic)"
        
    return {
        "duration_seconds": round(duration, 2),
        "total_frames": total_frames,
        "sampled_frames_count": sampled_count,
        "average_motion": round(avg_motion, 2),
        "average_brightness": round(avg_brightness, 2),
        "pacing_style": pacing,
        "overall_sentiment": overall_sentiment,
        "sentiment_distribution": {
            "Positive": pos_count,
            "Negative": neg_count,
            "Neutral": neu_count
        },
        "timeline": timeline
    }

def call_gemini_api(api_key: str, prompt: str, image_bytes_list: list, mime_types: list) -> dict:
    """Call Gemini REST API directly with prompt and base64 encoded images."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Construct standard Gemini API contents body
    parts = []
    
    # Add image data parts
    for img_bytes, mime_type in zip(image_bytes_list, mime_types):
        b64_data = base64.b64encode(img_bytes).decode('utf-8')
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": b64_data
            }
        })
        
    # Add final text prompt part
    parts.append({"text": prompt})
    
    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")
        
    try:
        response_json = response.json()
        content_text = response_json['candidates'][0]['content']['parts'][0]['text']
        # Remove markdown wrappers if any
        content_text = content_text.strip()
        if content_text.startswith("```json"):
            content_text = content_text[7:]
        if content_text.endswith("```"):
            content_text = content_text[:-3]
        content_text = content_text.strip()
        
        import json
        return json.loads(content_text)
    except Exception as e:
        raise Exception(f"Failed to parse Gemini response: {e}. Raw response: {response.text}")
