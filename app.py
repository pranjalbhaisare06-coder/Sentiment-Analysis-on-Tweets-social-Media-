import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import subprocess
import os
import socket
import sys

# Page Configuration
st.set_page_config(
    page_title="Sentira - Multimodal Sentiment & Media Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Glassmorphism & Vibrant Gradients)
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Font Overrides */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Main Layout styling */
        .main {
            background: radial-gradient(circle at 10% 20%, rgba(90, 18, 142, 0.08) 0%, rgba(2, 0, 15, 0.05) 90%);
        }
        
        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.07);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease-in-out;
        }
        .glass-card:hover {
            border-color: rgba(168, 85, 247, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px 0 rgba(168, 85, 247, 0.15);
        }
        
        /* Sentiment Badges */
        .sentiment-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 99px;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .badge-positive {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .badge-neutral {
            background: rgba(156, 163, 175, 0.15);
            color: #9ca3af;
            border: 1px solid rgba(156, 163, 175, 0.3);
        }
        
        /* Linear Gradient Headers */
        .gradient-text {
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Stats Styling */
        .stat-val {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 2px;
        }
        .stat-lbl {
            font-size: 0.85rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Image Color Swatches */
        .color-swatch {
            display: inline-block;
            width: 48px;
            height: 48px;
            border-radius: 8px;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }
        .color-swatch:hover {
            transform: scale(1.1);
        }
        
        /* File Upload Styling Override */
        .stFileUploader {
            border: 2px dashed rgba(168, 85, 247, 0.3) !important;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.01);
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Helper: Check if backend port is open
def is_backend_online(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# Helper: Automatically start the backend server
@st.cache_resource
def start_backend_server():
    if not is_backend_online(8000):
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
        # Start backend as background subprocess
        process = subprocess.Popen(
            cmd,
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give server 1.5 seconds to bind to port
        time.sleep(1.5)
        return True, process
    return False, None

# Initialize Backend
started, proc = start_backend_server()

# Sidebar Setup
st.sidebar.markdown("<h2 class='gradient-text'>⚡ SENTIRA SETTINGS</h2>", unsafe_allow_html=True)

# API status indicator
backend_online = is_backend_online(8000)
if backend_online:
    st.sidebar.success("🟢 API Server: Online (Port 8000)")
else:
    st.sidebar.error("🔴 API Server: Offline (Starting...)")
    # Trigger retry
    st.rerun()

st.sidebar.markdown("---")

# Gemini API Key setup
st.sidebar.subheader("🤖 Gemini API Integration")
gemini_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Enables advanced visual analysis for images & videos")
if gemini_key:
    st.sidebar.info("✨ Gemini Multimodal Engine Enabled")
else:
    st.sidebar.warning("⚠️ Using local visual heuristic engines")

st.sidebar.markdown("---")

# Session History management
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Clear Session Logs"):
    st.session_state.history = []
    st.session_state.chat_history = []
    if "active_tweets_result" in st.session_state:
        del st.session_state["active_tweets_result"]
    if "active_image_result" in st.session_state:
        del st.session_state["active_image_result"]
    if "active_video_result" in st.session_state:
        del st.session_state["active_video_result"]
    st.success("Session logs and chat history cleared!")
    st.rerun()

st.sidebar.markdown(
    """
    <div style='margin-top: 120px; font-size: 0.8rem; color: #6b7280; text-align: center;'>
        Sentira Dashboard v1.0.0<br>
        Developed for Tweets, Reels, Videos & Images
    </div>
    """, 
    unsafe_allow_html=True
)

# API Endpoint URL
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://sentiment-analysis-on-tweets-social-media.onrender.com"
)

# Main Layout
inject_custom_css()
st.markdown("<h1 style='margin-bottom: 0px;'>Sentira <span class='gradient-text'>Sentiment AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; font-size: 1.1rem;'>Cross-platform sentiment analysis dashboard for text, tweets, reels, videos, and images.</p>", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs(["🏠 OVERVIEW", "🐦 TWITTER & TEXT", "🖼️ IMAGE ANALYSIS", "🎥 REELS & VIDEO"])

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.subheader("Global Metrics & Activities")
    
    # Calculate global stats from session history
    hist_df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyzed = len(hist_df)
    positive_count = len(hist_df[hist_df['sentiment'] == 'Positive']) if total_analyzed else 0
    negative_count = len(hist_df[hist_df['sentiment'] == 'Negative']) if total_analyzed else 0
    neutral_count = len(hist_df[hist_df['sentiment'] == 'Neutral']) if total_analyzed else 0
    
    with col1:
        st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div class="stat-val">{total_analyzed}</div>
                <div class="stat-lbl">Total Items Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #10b981;">
                <div class="stat-val" style="color: #10b981;">{positive_count}</div>
                <div class="stat-lbl">Positive Detections</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #ef4444;">
                <div class="stat-val" style="color: #ef4444;">{negative_count}</div>
                <div class="stat-lbl">Negative Detections</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-left: 4px solid #9ca3af;">
                <div class="stat-val" style="color: #9ca3af;">{neutral_count}</div>
                <div class="stat-lbl">Neutral Detections</div>
            </div>
        """, unsafe_allow_html=True)

    if total_analyzed > 0:
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### Sentiment Distribution")
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [positive_count, neutral_count, negative_count]
            })
            st.bar_chart(chart_data.set_index("Sentiment"), color="#a855f7")
            
        with c2:
            st.markdown("#### Platforms Activity Breakdown")
            platform_counts = hist_df['platform'].value_counts().reset_index()
            platform_counts.columns = ['Platform', 'Count']
            st.bar_chart(platform_counts.set_index("Platform"), color="#3b82f6")
            
        st.markdown("#### Recent Operations Log")
        # Format logs for display
        display_logs = []
        for h in reversed(st.session_state.history):
            display_logs.append({
                "Timestamp": h["time"],
                "Platform": h["platform"],
                "Input Type": h["type"],
                "Source/Filename": h["source"],
                "Sentiment": h["sentiment"]
            })
        st.table(pd.DataFrame(display_logs))
    else:
        st.info("💡 No sentiment logs in current session. Upload images/videos or analyze tweets to build dashboard metrics!")

# --- TAB 2: TWITTER & TEXT ---
with tabs[1]:
    st.subheader("Tweet & Text Sentiment Analysis")
    
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.markdown("#### Input Source Selection")
        input_type = st.radio("Choose Input Type", ["Direct Text Input", "Fetch User Tweets"])
        
        if input_type == "Direct Text Input":
            text_input = st.text_area("Enter text payload", height=150, placeholder="Type a sentence or copy-paste tweet content here...")
            if st.button("Analyze Text Sentiment", use_container_width=True):
                if text_input.strip():
                    with st.spinner("Analyzing text..."):
                        response = requests.post(f"{API_BASE_URL}/analyze/text", json={"text": text_input})
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Add to session history
                            st.session_state.history.append({
                                "time": time.strftime("%H:%M:%S"),
                                "platform": "Text/Twitter",
                                "type": "Text",
                                "source": "Direct Input",
                                "sentiment": data["sentiment"]
                            })
                            
                            st.success("Analysis complete!")
                            
                            # Display individual results
                            badge_class = "badge-positive" if data["sentiment"] == "Positive" else ("badge-negative" if data["sentiment"] == "Negative" else "badge-neutral")
                            st.markdown(f"""
                                <div class="glass-card">
                                    <div class="sentiment-badge {badge_class}">{data["sentiment"]} Sentiment</div>
                                    <h3>Confidence Details</h3>
                                    <p><b>VADER Score:</b> {data['score']} (Range: -1.0 to +1.0)</p>
                                    <p><b>Local TF-IDF Model Sentiment:</b> {data['local_model_sentiment']}</p>
                                    <p style="color: #9ca3af; font-style: italic;">"{data['text']}"</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Error calling API: {response.text}")
                else:
                    st.warning("Please enter text first.")
                    
        elif input_type == "Fetch User Tweets":
            username = st.text_input("Twitter Username", placeholder="@jack")
            limit = st.slider("Number of tweets to retrieve", min_value=1, max_value=20, value=5)
            use_live = st.checkbox("Experimental Live Twitter Scrape (Requires Nitter)", value=False)
            
            if st.button("Fetch and Analyze Tweets", use_container_width=True):
                if username.strip():
                    with st.spinner("Connecting & analyzing tweets..."):
                        response = requests.post(
                            f"{API_BASE_URL}/analyze/tweets",
                            json={"username": username, "count": limit, "use_live": use_live}
                        )
                        if response.status_code == 200:
                            tweet_data = response.json()
                            
                            st.session_state["active_tweets_result"] = tweet_data
                            
                            # Batch add to history
                            for tw in tweet_data["tweets"]:
                                st.session_state.history.append({
                                    "time": time.strftime("%H:%M:%S"),
                                    "platform": "Twitter",
                                    "type": "Tweet",
                                    "source": f"@{username}",
                                    "sentiment": tw["sentiment"]
                                })
                            st.success(f"Successfully processed {tweet_data['count']} tweets!")
                            st.rerun()
                        else:
                            st.error(f"Error fetching tweets: {response.text}")
                else:
                    st.warning("Please enter a username.")
                    
    with col_out:
        st.markdown("#### Analysis Report")
        if "active_tweets_result" in st.session_state:
            res = st.session_state["active_tweets_result"]
            is_mock_str = "⚠️ Simulated Sandbox Mode" if res["is_mock"] else "🟢 Live Scraper Mode"
            
            st.markdown(f"""
                <div class="glass-card">
                    <h3>Analysis for @{res['username']}</h3>
                    <p style="color: #a855f7; font-weight: 600;">{is_mock_str}</p>
                    <div style="display: flex; gap: 20px; margin-top: 15px;">
                        <div>
                            <span style="color: #10b981; font-weight: bold; font-size: 1.5rem;">{res['metrics']['Positive']}</span> Positive
                        </div>
                        <div>
                            <span style="color: #ef4444; font-weight: bold; font-size: 1.5rem;">{res['metrics']['Negative']}</span> Negative
                        </div>
                        <div>
                            <span style="color: #9ca3af; font-weight: bold; font-size: 1.5rem;">{res['metrics']['Neutral']}</span> Neutral
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Processed Feed")
            for t in res["tweets"]:
                b_class = "badge-positive" if t["sentiment"] == "Positive" else ("badge-negative" if t["sentiment"] == "Negative" else "badge-neutral")
                st.markdown(f"""
                    <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="sentiment-badge {b_class}">{t['sentiment']}</span>
                            <span style="font-size: 0.8rem; color: #6b7280;">{t['date']}</span>
                        </div>
                        <p style="margin: 8px 0; font-size: 0.95rem;">{t['text']}</p>
                        <div style="font-size: 0.8rem; color: #a855f7;">
                            ❤️ {t['stats']['likes']} Likes &nbsp;&nbsp;&nbsp; 🔁 {t['stats']['retweets']} Retweets
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run an analysis on the left to see details here.")

# --- TAB 3: IMAGE ANALYSIS ---
with tabs[2]:
    st.subheader("Image Mood & Visual Sentiment Analyzer")
    
    col_img_in, col_img_out = st.columns([1, 1])
    
    with col_img_in:
        st.markdown("#### Upload Image")
        uploaded_image = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp", "bmp"])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image Preview", use_container_width=True)
            
            if st.button("Run Image Sentiment Analysis", use_container_width=True):
                with st.spinner("Processing image pixels and calling backend..."):
                    # Prepare file payload
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    data = {"gemini_api_key": gemini_key}
                    
                    response = requests.post(f"{API_BASE_URL}/analyze/image", files=files, data=data)
                    if response.status_code == 200:
                        st.session_state["active_image_result"] = response.json()
                        st.session_state["active_image_name"] = uploaded_image.name
                        
                        # Add to history
                        st.session_state.history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "platform": "Instagram / Image",
                            "type": "Image",
                            "source": uploaded_image.name,
                            "sentiment": response.json()["sentiment"]
                        })
                        st.success("Image analyzed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {response.text}")
                        
    with col_img_out:
        st.markdown("#### Visual Sentiment Report")
        if "active_image_result" in st.session_state:
            img_res = st.session_state["active_image_result"]
            img_name = st.session_state["active_image_name"]
            
            b_class = "badge-positive" if img_res["sentiment"] == "Positive" else ("badge-negative" if img_res["sentiment"] == "Negative" else "badge-neutral")
            
            st.markdown(f"""
                <div class="glass-card">
                    <span style="font-size: 0.8rem; color: #a855f7; text-transform: uppercase;">Method: {img_res.get('method', 'Local Heuristics')}</span>
                    <h3 style="margin-top: 5px; margin-bottom: 15px;">{img_name}</h3>
                    <div class="sentiment-badge {b_class}">{img_res['sentiment']} Sentiment</div>
                    <p><b>Confidence Score:</b> {img_res['confidence']}</p>
                    
                    <div class="glass-card" style="background: rgba(0,0,0,0.1); margin-top: 15px;">
                        <h5 style="margin-bottom: 8px;">Visual Mood Breakdown</h5>
                        <p style="font-size: 0.95rem; font-style: italic;">"{img_res['mood_description']}"</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show local heuristics detail
            heuristics = img_res.get("heuristics", img_res)
            
            st.markdown("##### Pixel Analytics")
            st.write(f"**Resolution**: {heuristics.get('dimensions', 'N/A')}")
            
            col_b, col_c = st.columns(2)
            with col_b:
                st.metric("Avg Brightness (0-255)", heuristics.get("brightness", 0.0))
            with col_c:
                st.metric("Contrast (Variance)", heuristics.get("contrast", 0.0))
                
            # Render color swatches
            st.markdown("##### Dominant Colors Detected")
            swatches_html = ""
            for hex_color in heuristics.get("dominant_colors", []):
                swatches_html += f'<div class="color-swatch" style="background-color: {hex_color};" title="{hex_color}"></div>'
            st.markdown(swatches_html, unsafe_allow_html=True)
        else:
            st.info("Upload and analyze an image to view pixel color histograms and mood analytics.")

# --- TAB 4: REELS & VIDEO ---
with tabs[3]:
    st.subheader("Instagram Reels & Video Sentiment Analyzer")
    
    col_vid_in, col_vid_out = st.columns([1, 1])
    
    with col_vid_in:
        st.markdown("#### Upload Reel / Video")
        uploaded_video = st.file_uploader("Upload video file...", type=["mp4", "avi", "mov", "webm", "mkv"])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("Run Video Timeline Analysis", use_container_width=True):
                with st.spinner("Processing video frames (this may take a few seconds)..."):
                    files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                    data = {"gemini_api_key": gemini_key}
                    
                    response = requests.post(f"{API_BASE_URL}/analyze/video", files=files, data=data)
                    if response.status_code == 200:
                        st.session_state["active_video_result"] = response.json()
                        st.session_state["active_video_name"] = uploaded_video.name
                        
                        # Add to history
                        st.session_state.history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "platform": "Instagram / Reel",
                            "type": "Video",
                            "source": uploaded_video.name,
                            "sentiment": response.json()["sentiment"]
                        })
                        st.success("Video analysis completed!")
                        st.rerun()
                    else:
                        st.error(f"Video analysis failed: {response.text}")
                        
    with col_vid_out:
        st.markdown("#### Video Analysis Report")
        if "active_video_result" in st.session_state:
            vid_res = st.session_state["active_video_result"]
            vid_name = st.session_state["active_video_name"]
            
            b_class = "badge-positive" if vid_res["sentiment"] == "Positive" else ("badge-negative" if vid_res["sentiment"] == "Negative" else "badge-neutral")
            
            st.markdown(f"""
                <div class="glass-card">
                    <span style="font-size: 0.8rem; color: #a855f7; text-transform: uppercase;">Method: {vid_res.get('method', 'Local Heuristics')}</span>
                    <h3 style="margin-top: 5px; margin-bottom: 15px;">{vid_name}</h3>
                    <div class="sentiment-badge {b_class}">{vid_res['sentiment']} Sentiment</div>
                    <p><b>Confidence Rating:</b> {vid_res.get('confidence', 0.80)}</p>
                    <p><b>Video Duration:</b> {vid_res.get('heuristics', vid_res).get('duration_seconds', 0.0)} seconds</p>
                    <p><b>Motion / Pacing Style:</b> {vid_res.get('heuristics', vid_res).get('pacing_style', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show Gemini Explanation if available
            if "explanation" in vid_res:
                st.markdown("##### AI Multimodal Scene Analysis")
                st.info(vid_res["explanation"])
                
            # Render visual dynamics chart from heuristics
            heuristics = vid_res.get("heuristics", vid_res)
            timeline = heuristics.get("timeline", [])
            
            if timeline:
                st.markdown("##### Visual Timeline Dynamics (Motion vs Brightness)")
                time_df = pd.DataFrame(timeline)
                # Plot motion and brightness over time
                chart_df = time_df[["timestamp", "motion", "brightness"]].set_index("timestamp")
                st.line_chart(chart_df)
                
                # Show Gemini timeline if available
                if "gemini_timeline" in vid_res:
                    st.markdown("##### Sequential AI Narrative Timeline")
                    for frame_info in vid_res["gemini_timeline"]:
                        st.markdown(f"**Index {frame_info.get('frame_index') or frame_info.get('second')}s** - *{frame_info.get('sentiment')}*: {frame_info.get('description') or frame_info.get('explanation')}")
            else:
                st.warning("No timeline data generated.")
        else:
            st.info("Upload and analyze a video to see motion dynamics, pacing styles, and timeline charts.")