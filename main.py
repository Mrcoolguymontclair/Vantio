print(">>> VANTIO v4.1 - AI-FIRST EDITION <<<")
import streamlit as st
import pandas as pd
import altair as alt
import requests
import dateutil.parser
import isodate
import httpx 
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import re
import json
import time
from typing import Dict, List, Optional
from PIL import Image
from io import BytesIO
import colorsys

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Vantio AI - YouTube Intelligence", 
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': "Vantio v4.1 - AI-First YouTube Analytics"}
)

# Load Secrets
try:
    YOUTUBE_KEY = st.secrets["YOUTUBE_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error(f"🚨 Configuration Error: {e}")
    st.stop()

# AI Client
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_KEY, http_client=httpx.Client())
    ai_available = True
except:
    ai_available = False

# --- UTILITY FUNCTIONS ---
def extract_dominant_color(image_url):
    """Extract dominant color from image"""
    try:
        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content))
        img = img.resize((50, 50))
        img = img.convert('RGB')

        pixels = list(img.getdata())

        # Filter out very dark and very light colors
        filtered = [p for p in pixels if sum(p) > 100 and sum(p) < 650]

        if not filtered:
            filtered = pixels

        # Get most common color
        color_counter = Counter(filtered)
        dominant = color_counter.most_common(1)[0][0]

        # Convert to hex and enhance saturation
        r, g, b = dominant
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        s = min(1.0, s * 1.5)  # Boost saturation
        v = min(1.0, v * 1.2)  # Boost brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    except:
        return "#FF4B4B"

def format_number_str(num):
    """Format numbers with K, M, B suffixes"""
    try:
        num = int(num)
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return f"{num:,}"
    except:
        return str(num)

def parse_duration(dur_str):
    try: 
        return isodate.parse_duration(dur_str).total_seconds() / 60
    except: 
        return 0

def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', text.lower())
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 
                  'your', 'about', 'what', 'when', 'where', 'make', 'more',
                  'best', 'most', 'does', 'work', 'need', 'know'}
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(filtered).most_common(top_n)

def calculate_growth_rate(df):
    if len(df) < 5: 
        return 0
    df_sorted = df.sort_values('Publish Date')
    recent_count = min(15, len(df) // 3)
    older_count = min(15, len(df) // 3)
    recent = df_sorted.tail(recent_count)['Views'].mean()
    older = df_sorted.head(older_count)['Views'].mean()
    return ((recent - older) / older * 100) if older > 0 else 0

def find_optimal_posting_time(df):
    if df.empty or len(df) < 10: 
        return None
    day_hour_stats = df.groupby(['Day', 'Hour']).agg({
        'Views': ['mean', 'count'],
        'Engagement Rate': 'mean'
    }).reset_index()
    day_hour_stats.columns = ['Day', 'Hour', 'Avg_Views', 'Count', 'Avg_Engagement']
    significant = day_hour_stats[day_hour_stats['Count'] >= 2]
    if significant.empty:
        return None
    best = significant.nlargest(1, 'Avg_Views').iloc[0]
    return f"{best['Day']} at {int(best['Hour'])}:00"

def calculate_consistency_score(df):
    """Measure upload consistency with precision fixes"""
    if len(df) < 3: 
        return 0

    df_sorted = df.sort_values('Publish Date')
    gaps = df_sorted['Publish Date'].diff().dt.total_seconds().dropna() / 86400
    gaps = gaps[gaps > 0.04]

    if gaps.empty or gaps.mean() == 0: 
        return 0

    cv = gaps.std() / gaps.mean()
    consistency = max(0, min(100, 100 - (cv * 30)))

    return consistency

def predict_next_video_performance(df):
    if len(df) < 5: 
        return None
    df_sorted = df.sort_values('Publish Date')
    recent_10 = df_sorted.tail(10)['Views'].values
    weights = np.linspace(0.5, 1.5, len(recent_10))
    weighted_avg = np.average(recent_10, weights=weights)
    overall_avg = df['Views'].mean()
    prediction = int(weighted_avg * 0.7 + overall_avg * 0.3)
    return prediction

def calculate_content_diversity_score(df):
    if df.empty:
        return 0
    all_titles = ' '.join(df['Title'].tolist())
    keywords = extract_keywords(all_titles, top_n=50)
    if not keywords:
        return 0
    total_words = sum(count for _, count in keywords)
    entropy = -sum((count/total_words) * np.log(count/total_words) for _, count in keywords if count > 0)
    max_entropy = np.log(len(keywords))
    diversity = (entropy / max_entropy * 100) if max_entropy > 0 else 0
    return min(100, diversity)

def calculate_monetization_potential(df, subscriber_count):
    if df.empty:
        return {}
    avg_views = df['Views'].mean()
    low_rpm, avg_rpm, high_rpm = 1.5, 3.5, 8.0
    df_sorted = df.sort_values('Publish Date', ascending=False)
    monthly_videos = min(len(df_sorted), max(1, len(df_sorted) // 6))
    return {
        'est_monthly_revenue_low': int((avg_views * monthly_videos * low_rpm) / 1000),
        'est_monthly_revenue_avg': int((avg_views * monthly_videos * avg_rpm) / 1000),
        'est_monthly_revenue_high': int((avg_views * monthly_videos * high_rpm) / 1000),
        'monthly_video_count': monthly_videos,
        'avg_views_per_video': int(avg_views)
    }

@st.cache_data(ttl=3600, show_spinner=False)
def search_channel(query):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet", 
        "q": query, 
        "type": "channel", 
        "maxResults": 5, 
        "order": "relevance", 
        "key": YOUTUBE_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if "items" not in res or not res["items"]: 
            return None
        item = res['items'][0]
        return {
            'id': item['snippet']['channelId'],
            'title': item['snippet']['title'],
            'thumbnail': item['snippet']['thumbnails']['high']['url'],
            'description': item['snippet']['description']
        }
    except Exception as e:
        st.error(f"Search error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_channel_statistics(channel_id):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        "part": "statistics,contentDetails,snippet,brandingSettings", 
        "id": channel_id, 
        "key": YOUTUBE_KEY
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if "items" not in res: 
            return None
        item = res['items'][0]
        return {
            'stats': item['statistics'],
            'uploads_id': item['contentDetails']['relatedPlaylists']['uploads'],
            'created_date': item['snippet']['publishedAt'],
            'country': item['snippet'].get('country', 'Unknown'),
            'custom_url': item['snippet'].get('customUrl', ''),
            'keywords': item.get('brandingSettings', {}).get('channel', {}).get('keywords', '')
        }
    except Exception as e:
        st.error(f"Statistics error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_videos(uploads_id, max_results=50):
    video_items = []
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    next_page_token = None
    for _ in range(3):
        params = {
            "part": "snippet,contentDetails", 
            "playlistId": uploads_id,
            "maxResults": 50, 
            "key": YOUTUBE_KEY
        }
        if next_page_token:
            params['pageToken'] = next_page_token
        try:
            res = requests.get(base_url, params=params, timeout=10).json()
            if "items" in res:
                video_items.extend(res['items'])
            next_page_token = res.get('nextPageToken')
            if not next_page_token:
                break
        except:
            break
    video_ids = [item['contentDetails']['videoId'] for item in video_items]
    final_data = []
    if video_ids:
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            vid_url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "statistics,snippet,contentDetails", 
                "id": ",".join(batch), 
                "key": YOUTUBE_KEY
            }
            try:
                res = requests.get(vid_url, params=params, timeout=10).json()
                if "items" in res:
                    for v in res['items']:
                        stats = v.get('statistics', {})
                        views = int(stats.get('viewCount', 0))
                        likes = int(stats.get('likeCount', 0))
                        comments = int(stats.get('commentCount', 0))
                        duration = parse_duration(v['contentDetails']['duration'])
                        try: 
                            pub_date = dateutil.parser.parse(v['snippet']['publishedAt'])
                        except: 
                            pub_date = datetime.now()
                        eng_rate = ((likes + comments) / views * 100) if views > 0 else 0
                        final_data.append({
                            "Title": v['snippet']['title'], 
                            "Publish Date": pub_date,
                            "Day": pub_date.strftime("%A"), 
                            "Hour": pub_date.hour, 
                            "Duration (Mins)": round(duration, 1),
                            "Views": views,
                            "Likes": likes,
                            "Comments": comments,
                            "Engagement Rate": round(eng_rate, 2),
                            "Thumbnail": v['snippet']['thumbnails']['medium']['url'],
                            "URL": f"https://www.youtube.com/watch?v={v['id']}",
                            "Video ID": v['id'],
                            "Description": v['snippet'].get('description', '')[:200]
                        })
            except:
                continue
    return final_data

def stream_ai_response(prompt, model="llama-3.1-8b-instant"):
    if not ai_available:
        yield "⚠️ AI service temporarily unavailable."
        return
    try:
        stream = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert YouTube growth strategist. Provide specific, actionable advice with concrete examples."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content: 
                yield chunk.choices[0].delta.content
    except Exception as e: 
        yield f"⚠️ AI Error: {str(e)}"

# --- SESSION STATE ---
if 'report_data' not in st.session_state: 
    st.session_state['report_data'] = None
if 'chat_history' not in st.session_state: 
    st.session_state['chat_history'] = []
if 'theme_color' not in st.session_state: 
    st.session_state['theme_color'] = "#FF4B4B"
if 'analysis_history' not in st.session_state: 
    st.session_state['analysis_history'] = []
if 'light_mode' not in st.session_state:
    st.session_state['light_mode'] = False
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "chat"
if 'pending_question' not in st.session_state:
    st.session_state['pending_question'] = None

# --- DYNAMIC STYLING ---
def get_styles():
    color = st.session_state['theme_color']
    is_light = st.session_state['light_mode']

    if is_light:
        bg_colors = f"#ffffff, #f8f9fa, #f1f3f5, {color}15"
        text_color = "#000000"
        secondary_text = "#495057"
        card_bg = "rgba(0, 0, 0, 0.03)"
        card_hover = "rgba(0, 0, 0, 0.05)"
        border_color = "rgba(0, 0, 0, 0.1)"
        metric_value_color = "#000000"
    else:
        bg_colors = f"#000000, #0a0a0a, #1a1a1a, {color}15"
        text_color = "#ffffff"
        secondary_text = "#e0e0e0"
        card_bg = "rgba(255, 255, 255, 0.04)"
        card_hover = "rgba(255, 255, 255, 0.08)"
        border_color = "rgba(255, 255, 255, 0.1)"
        metric_value_color = "#ffffff"

    return f"""
<style>
    @keyframes gradient-bg {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .stApp {{
        background: linear-gradient(-45deg, {bg_colors});
        background-size: 400% 400%;
        animation: gradient-bg 20s ease infinite;
    }}

    @keyframes shine {{
        to {{ background-position: 200% center; }}
    }}

    .main-title {{
        font-size: 6rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: -4px;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(to right, {color} 20%, {text_color} 50%, {color} 80%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 80px {color}40;
    }}

    .subtitle {{
        text-align: center;
        color: {secondary_text};
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }}

    /* Enhanced Metrics */
    .stMetric {{
        background: {card_bg} !important;
        backdrop-filter: blur(15px);
        border: 1px solid {border_color} !important;
        border-radius: 16px;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }}

    .stMetric:hover {{
        background: {card_hover} !important;
        border-color: {color}40 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 32px {color}20;
    }}

    .stMetric label {{
        color: {secondary_text} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }}

    .stMetric [data-testid="stMetricValue"] {{
        color: {metric_value_color} !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }}

    .stMetric [data-testid="stMetricDelta"] {{
        white-space: nowrap !important;
        overflow: visible !important;
    }}

    .insight-card {{
        background: linear-gradient(135deg, {card_bg} 0%, {card_hover} 100%);
        border-left: 4px solid {color};
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: {text_color} !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}

    .insight-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 8px 24px {color}30;
    }}

    h1, h2, h3, h4 {{ 
        color: {text_color} !important; 
        font-weight: 700 !important;
    }}

    p, label, .stMarkdown, div[data-testid="stCaptionContainer"] {{ 
        color: {text_color} !important; 
    }}

    /* Chat styling */
    .stChatMessage {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}

    /* Buttons */
    button[kind="primary"] {{
        background: linear-gradient(135deg, {color} 0%, {color}dd 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px {color}40 !important;
    }}

    button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px {color}60 !important;
    }}

    button[kind="secondary"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }}

    button[kind="secondary"]:hover {{
        background: {card_hover} !important;
        border-color: {color}40 !important;
        transform: translateY(-1px) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {card_bg};
        padding: 0.5rem;
        border-radius: 12px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: {text_color};
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        background: {color} !important;
        color: white !important;
    }}

    /* Sidebar label */
    button[data-testid="collapsedControl"] {{
        position: relative;
    }}

    button[data-testid="collapsedControl"]::after {{
        content: "📜 History";
        position: absolute;
        left: 50px;
        top: 50%;
        transform: translateY(-50%);
        background: {color};
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
        pointer-events: none;
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}

    /* Expander */
    div[data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}

    /* Input fields */
    .stTextInput > div > div > input {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        border-radius: 8px !important;
    }}

    /* Select boxes */
    .stSelectbox > div > div {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
        border-radius: 8px !important;
    }}

    /* Status indicator */
    .stStatus {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
    }}
</style>
"""

st.markdown(get_styles(), unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 1.5rem;'>🚀 Vantio AI</h2>", unsafe_allow_html=True)

    # Theme toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Light" if not st.session_state['light_mode'] else "🌙 Dark", use_container_width=True):
            st.session_state['light_mode'] = not st.session_state['light_mode']
            st.rerun()

    with col2:
        picked_color = st.color_picker("🎨", st.session_state['theme_color'], label_visibility="collapsed")
        if picked_color != st.session_state['theme_color']:
            st.session_state['theme_color'] = picked_color
            st.rerun()

    st.divider()

    # Analysis History
    if st.session_state['analysis_history']:
        st.subheader("📊 Recent Analyses")
        for idx, channel in enumerate(reversed(st.session_state['analysis_history'][-5:])):
            if st.button(f"🔄 {channel}", key=f"history_{idx}", use_container_width=True):
                st.session_state['report_data'] = None
                st.session_state['chat_history'] = []
                st.rerun()

    st.divider()

    if st.button("🔄 New Analysis", type="primary", use_container_width=True):
        st.session_state['report_data'] = None
        st.session_state['chat_history'] = []
        st.rerun()

    st.markdown("---")
    st.caption("Vantio v4.1 AI-First Edition")
    st.caption("Powered by Claude AI & Groq")

# --- MAIN APPLICATION ---
if st.session_state['report_data'] is None:
    # Landing Page
    st.write("")
    st.markdown("<div class='main-title'>VANTIO AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-POWERED YOUTUBE GROWTH PLATFORM</div>", unsafe_allow_html=True)

    st.write("")

    # Search interface
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("search_form"):
            channel_query = st.text_input(
                "Enter Channel Name or URL", 
                placeholder="e.g., MrBeast, MKBHD, or paste URL",
                label_visibility="collapsed"
            )

            submit = st.form_submit_button("🔍 Analyze with AI", type="primary", use_container_width=True)

            if submit and channel_query:
                with st.status("🔍 AI scanning YouTube...", expanded=True) as status:
                    st.write("📡 Finding channel...")
                    channel_info = search_channel(channel_query)

                    if channel_info:
                        st.write("🎨 Extracting brand colors...")
                        dominant_color = extract_dominant_color(channel_info['thumbnail'])
                        st.session_state['theme_color'] = dominant_color

                        st.write("📊 Gathering analytics...")
                        stats_blob = get_channel_statistics(channel_info['id'])

                        if stats_blob:
                            st.write("🎥 Analyzing videos...")
                            raw_videos = fetch_all_videos(stats_blob['uploads_id'], max_results=150)

                            st.session_state['report_data'] = {
                                "name": channel_info['title'], 
                                "thumbnail": channel_info['thumbnail'],
                                "description": channel_info.get('description', ''),
                                "subs": int(stats_blob['stats']['subscriberCount']),
                                "total_views": int(stats_blob['stats']['viewCount']),
                                "total_videos": int(stats_blob['stats']['videoCount']),
                                "created_date": stats_blob['created_date'],
                                "country": stats_blob.get('country', 'Unknown'),
                                "raw_videos": raw_videos
                            }

                            if channel_info['title'] not in st.session_state['analysis_history']:
                                st.session_state['analysis_history'].append(channel_info['title'])

                            st.session_state['chat_history'] = []
                            st.session_state['active_tab'] = "chat"
                            status.update(label="✅ Analysis Complete!", state="complete")
                            st.rerun()
                        else:
                            status.update(label="❌ Could not fetch statistics", state="error")
                    else:
                        status.update(label="❌ Channel Not Found", state="error")

    st.write("")
    st.write("")

    # Feature showcase
    feat1, feat2, feat3, feat4 = st.columns(4)

    with feat1:
        st.markdown("""
        <div class='insight-card'>
        <h4>🤖 AI Assistant</h4>
        <p style='font-size: 0.9rem;'>Chat with AI for personalized growth strategies</p>
        </div>
        """, unsafe_allow_html=True)

    with feat2:
        st.markdown("""
        <div class='insight-card'>
        <h4>📊 Deep Analytics</h4>
        <p style='font-size: 0.9rem;'>Comprehensive performance metrics and insights</p>
        </div>
        """, unsafe_allow_html=True)

    with feat3:
        st.markdown("""
        <div class='insight-card'>
        <h4>🔮 Predictions</h4>
        <p style='font-size: 0.9rem;'>ML-based forecasting and trend analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with feat4:
        st.markdown("""
        <div class='insight-card'>
        <h4>💰 Revenue Intel</h4>
        <p style='font-size: 0.9rem;'>Monetization estimates and optimization</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # MAIN DASHBOARD - AI FIRST
    data = st.session_state['report_data']
    df = pd.DataFrame(data['raw_videos'])

    if not df.empty:
        # Calculate metrics
        growth_rate = calculate_growth_rate(df)
        consistency_score = calculate_consistency_score(df)
        diversity_score = calculate_content_diversity_score(df)
        monetization_data = calculate_monetization_potential(df, data['subs'])

        # Header with channel info
        header_col1, header_col2 = st.columns([1, 6])

        with header_col1:
            st.image(data['thumbnail'], width=120)

        with header_col2:
            st.title(data['name'])

            metric_cols = st.columns(6)
            metric_cols[0].metric("👥 Subscribers", format_number_str(data['subs']))
            metric_cols[1].metric("🎥 Videos", format_number_str(data['total_videos']))
            metric_cols[2].metric("👁️ Total Views", format_number_str(data['total_views']))
            metric_cols[3].metric("📈 Growth", f"{growth_rate:+.1f}%")
            metric_cols[4].metric("💬 Avg Engagement", f"{df['Engagement Rate'].mean():.2f}%")
            metric_cols[5].metric("🎯 Consistency", f"{consistency_score:.0f}/100")

        st.markdown("---")

        # Main tabs - AI FIRST
        tab_titles = ["🤖 AI Chat", "📊 Quick Insights", "📈 Analytics Deep Dive", "💰 Revenue", "🎥 Videos"]
        tabs = st.tabs(tab_titles)

        # ==================== TAB 1: AI CHAT (PRIMARY) ====================
        with tabs[0]:
            st.markdown("### 💬 AI Growth Strategist")

            # Suggested questions
            st.markdown("**💡 Quick Questions:**")
            quick_q_cols = st.columns(3)

            with quick_q_cols[0]:
                if st.button("🎯 What should I improve?", key="q1", use_container_width=True):
                    st.session_state['pending_question'] = "Based on my channel's data, what are the top 3 things I should improve immediately to grow faster?"

            with quick_q_cols[1]:
                if st.button("🔥 Viral video formula?", key="q2", use_container_width=True):
                    st.session_state['pending_question'] = "Analyze my top performing videos and tell me the formula for creating viral content on my channel."

            with quick_q_cols[2]:
                if st.button("📅 Best posting schedule?", key="q3", use_container_width=True):
                    st.session_state['pending_question'] = "What's the optimal posting schedule for my channel based on performance data?"

            st.markdown("---")

            # Chat messages
            chat_container = st.container(height=500)

            with chat_container:
                if not st.session_state['chat_history']:
                    st.info("👋 Hi! I'm your AI YouTube strategist. Ask me anything about growing your channel!")

                for msg in st.session_state['chat_history']:
                    with st.chat_message(msg["role"]): 
                        st.write(msg["content"])

            # Process pending question from button
            if st.session_state['pending_question']:
                user_input = st.session_state['pending_question']
                st.session_state['pending_question'] = None

                st.session_state['chat_history'].append({"role": "user", "content": user_input})

                with chat_container:
                    with st.chat_message("user"): 
                        st.write(user_input)

                    # Build context
                    top_vids = df.nlargest(5, 'Views')[['Title', 'Views']].to_string(index=False)
                    keywords = extract_keywords(' '.join(df['Title'].tolist()), top_n=10)

                    prompt = f"""You are an elite YouTube growth strategist. Analyze this channel and answer the user's question with specific, actionable advice.

CHANNEL: {data['name']}
SUBSCRIBERS: {format_number_str(data['subs'])}
VIDEOS ANALYZED: {len(df)}

KEY METRICS:
- Growth Rate: {growth_rate:.1f}%
- Avg Engagement: {df['Engagement Rate'].mean():.2f}%
- Consistency: {consistency_score:.0f}/100
- Diversity: {diversity_score:.0f}/100

TOP 5 VIDEOS:
{top_vids}

TOP KEYWORDS: {', '.join([w for w, c in keywords])}

USER QUESTION: {user_input}

Provide specific, data-driven advice with concrete examples and action steps."""

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_response = ""

                        for chunk in stream_ai_response(prompt):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")

                        placeholder.markdown(full_response)

                st.session_state['chat_history'].append({"role": "assistant", "content": full_response})
                st.rerun()

            # Chat input
            if user_input := st.chat_input("Ask your AI strategist anything..."):
                st.session_state['chat_history'].append({"role": "user", "content": user_input})

                with chat_container:
                    with st.chat_message("user"): 
                        st.write(user_input)

                    # Build context
                    top_vids = df.nlargest(5, 'Views')[['Title', 'Views']].to_string(index=False)
                    keywords = extract_keywords(' '.join(df['Title'].tolist()), top_n=10)

                    prompt = f"""You are an elite YouTube growth strategist. Analyze this channel and answer the user's question with specific, actionable advice.

CHANNEL: {data['name']}
SUBSCRIBERS: {format_number_str(data['subs'])}
VIDEOS ANALYZED: {len(df)}

KEY METRICS:
- Growth Rate: {growth_rate:.1f}%
- Avg Engagement: {df['Engagement Rate'].mean():.2f}%
- Consistency: {consistency_score:.0f}/100
- Diversity: {diversity_score:.0f}/100

TOP 5 VIDEOS:
{top_vids}

TOP KEYWORDS: {', '.join([w for w, c in keywords])}

USER QUESTION: {user_input}

Provide specific, data-driven advice with concrete examples and action steps."""

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_response = ""

                        for chunk in stream_ai_response(prompt):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")

                        placeholder.markdown(full_response)

                st.session_state['chat_history'].append({"role": "assistant", "content": full_response})
                st.rerun()

        # ==================== TAB 2: QUICK INSIGHTS ====================
        with tabs[1]:
            st.markdown("### ⚡ At-a-Glance Performance")

            # Key metrics grid
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("📊 Avg Views", format_number_str(df['Views'].mean()))
            kpi_cols[1].metric("⏱️ Avg Duration", f"{df['Duration (Mins)'].mean():.1f} min")
            kpi_cols[2].metric("🌈 Content Variety", f"{diversity_score:.0f}/100")
            kpi_cols[3].metric("💵 Est. Monthly", f"${monetization_data['est_monthly_revenue_avg']:,}")

            st.write("")

            # Visual summary
            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                st.markdown("#### 📈 Performance Trend")

                chart_df = df.sort_values('Publish Date').tail(30).copy()

                trend_chart = alt.Chart(chart_df).mark_line(
                    color=st.session_state['theme_color'],
                    strokeWidth=3,
                    point=True
                ).encode(
                    x=alt.X('Publish Date:T', title='Date'),
                    y=alt.Y('Views:Q', title='Views', scale=alt.Scale(type='log')),
                    tooltip=['Title', alt.Tooltip('Views:Q', format=','), alt.Tooltip('Publish Date:T', format='%b %d')]
                ).properties(height=300).interactive()

                st.altair_chart(trend_chart, use_container_width=True)

            with vis_col2:
                st.markdown("#### 🎯 Best Posting Days")

                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_stats = df.groupby('Day')['Views'].mean().reindex(day_order)

                day_data = pd.DataFrame({
                    'Day': day_stats.index,
                    'Avg Views': day_stats.values
                })

                day_chart = alt.Chart(day_data).mark_bar(
                    color=st.session_state['theme_color'],
                    opacity=0.8
                ).encode(
                    x=alt.X('Avg Views:Q', title='Average Views'),
                    y=alt.Y('Day:N', sort=day_order, title=None),
                    tooltip=['Day', alt.Tooltip('Avg Views:Q', format=',')]
                ).properties(height=300)

                st.altair_chart(day_chart, use_container_width=True)

            st.markdown("---")

            # Smart recommendations
            st.markdown("### 🎯 Smart Recommendations")

            rec_cols = st.columns(2)

            with rec_cols[0]:
                st.markdown("""
                <div class='insight-card'>
                <h4>⚡ Quick Wins</h4>
                </div>
                """, unsafe_allow_html=True)

                optimal_time = find_optimal_posting_time(df)
                if optimal_time:
                    st.markdown(f"✅ **Best time to post:** {optimal_time}")

                best_duration = df.nlargest(10, 'Views')['Duration (Mins)'].mean()
                st.markdown(f"✅ **Optimal video length:** {best_duration:.1f} minutes")

                if growth_rate > 15:
                    st.markdown("✅ **Strong momentum:** Increase upload frequency")
                elif growth_rate < 0:
                    st.markdown("⚠️ **Refresh strategy:** Analyze top performers")

            with rec_cols[1]:
                st.markdown("""
                <div class='insight-card'>
                <h4>🔥 Content Focus</h4>
                </div>
                """, unsafe_allow_html=True)

                keywords = extract_keywords(' '.join(df['Title'].tolist()), top_n=5)
                st.markdown("**Your winning topics:**")
                for word, count in keywords:
                    st.markdown(f"• **{word.title()}** ({count} videos)")

        # ==================== TAB 3: ANALYTICS DEEP DIVE ====================
        with tabs[2]:
            st.markdown("### 📊 Comprehensive Analytics")

            # Detailed metrics
            detail_cols = st.columns(5)
            detail_cols[0].metric("📈 Growth Rate", f"{growth_rate:+.1f}%")
            detail_cols[1].metric("🎯 Consistency", f"{consistency_score:.0f}/100")
            detail_cols[2].metric("🌈 Diversity", f"{diversity_score:.0f}/100")
            detail_cols[3].metric("👍 Avg Likes", format_number_str(df['Likes'].mean()))
            detail_cols[4].metric("💭 Avg Comments", format_number_str(df['Comments'].mean()))

            st.write("")

            # Performance timeline
            st.markdown("#### 📈 Complete Performance Timeline")

            timeline_df = df.sort_values('Publish Date').copy()

            base = alt.Chart(timeline_df).encode(
                x=alt.X('Publish Date:T', title='Publication Date'),
                tooltip=[
                    alt.Tooltip('Title:N', title='Video'),
                    alt.Tooltip('Views:Q', format=','),
                    alt.Tooltip('Engagement Rate:Q', format='.2f', title='Engagement %'),
                    alt.Tooltip('Publish Date:T', format='%B %d, %Y')
                ]
            )

            area = base.mark_area(
                line={'color': st.session_state['theme_color']},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[
                        alt.GradientStop(color=st.session_state['theme_color'], offset=0),
                        alt.GradientStop(color=st.session_state['theme_color'] + '00', offset=1)
                    ],
                    x1=1, x2=1, y1=1, y2=0
                ),
                opacity=0.4
            ).encode(y=alt.Y('Views:Q', title='Views'))

            line = base.mark_line(
                color=st.session_state['theme_color'],
                strokeWidth=2
            ).encode(y='Views:Q')

            combined = (area + line).properties(height=400).interactive()
            st.altair_chart(combined, use_container_width=True)

            st.markdown("---")

            # Top performers
            st.markdown("### 🏆 Top 10 Performing Videos")

            top_10 = df.nlargest(10, 'Views')

            for idx, (_, video) in enumerate(top_10.iterrows(), 1):
                with st.expander(f"{'🥇' if idx==1 else '🥈' if idx==2 else '🥉' if idx==3 else '📹'} {video['Title'][:80]}"):
                    vid_col1, vid_col2 = st.columns([1, 2])

                    with vid_col1:
                        st.image(video['Thumbnail'])

                    with vid_col2:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Views", format_number_str(video['Views']))
                        m2.metric("Likes", format_number_str(video['Likes']))
                        m3.metric("Engagement", f"{video['Engagement Rate']:.2f}%")
                        m4.metric("Duration", f"{video['Duration (Mins)']} min")

                        st.markdown(f"**Published:** {video['Publish Date'].strftime('%B %d, %Y')}")
                        st.markdown(f"[🔗 Watch Video]({video['URL']})")

        # ==================== TAB 4: REVENUE ====================
        with tabs[3]:
            st.markdown("### 💰 Revenue Intelligence")

            st.warning("⚠️ Estimates based on industry averages. Actual earnings vary by niche and audience.")

            rev_cols = st.columns(4)
            rev_cols[0].metric("💵 Monthly (Low)", f"${monetization_data['est_monthly_revenue_low']:,}")
            rev_cols[1].metric("💰 Monthly (Avg)", f"${monetization_data['est_monthly_revenue_avg']:,}")
            rev_cols[2].metric("🤑 Monthly (High)", f"${monetization_data['est_monthly_revenue_high']:,}")
            rev_cols[3].metric("📅 Annual (Avg)", f"${monetization_data['est_monthly_revenue_avg']*12:,}")

            st.write("")

            # Growth scenarios
            st.markdown("### 📈 Growth Scenarios")

            scenarios = []
            current_monthly = monetization_data['monthly_video_count']
            current_views = monetization_data['avg_views_per_video']

            for mult, label in [(1.0, "Current"), (1.5, "+50%"), (2.0, "2x"), (3.0, "3x")]:
                proj = (current_views * mult * current_monthly * 3.5) / 1000
                scenarios.append({"Scenario": label, "Monthly Revenue": int(proj)})

            scenario_df = pd.DataFrame(scenarios)

            scenario_chart = alt.Chart(scenario_df).mark_bar(
                color=st.session_state['theme_color'],
                opacity=0.8
            ).encode(
                x=alt.X('Scenario:N', sort=None, title=None),
                y=alt.Y('Monthly Revenue:Q', title='Est. Monthly Revenue ($)'),
                tooltip=['Scenario', alt.Tooltip('Monthly Revenue:Q', format='$,')]
            ).properties(height=350)

            st.altair_chart(scenario_chart, use_container_width=True)

            st.markdown("---")

            # Optimization tips
            st.markdown("### 💎 Monetization Tips")

            tip_cols = st.columns(2)

            with tip_cols[0]:
                st.markdown("""
                **🎯 Increase RPM:**
                - Target high-CPM topics (finance, tech, business)
                - Improve watch time for more ad impressions
                - Enable all ad formats
                - Create 10-15 min videos (optimal for ads)
                """)

            with tip_cols[1]:
                st.markdown("""
                **💰 Additional Streams:**
                - Channel memberships
                - Super Thanks/Super Chat
                - Merchandise shelf
                - Affiliate marketing
                - Brand sponsorships
                """)

        # ==================== TAB 5: VIDEOS ====================
        with tabs[4]:
            st.markdown("### 🎥 Video Library")

            # Filters
            filter_cols = st.columns(3)

            with filter_cols[0]:
                sort_by = st.selectbox("Sort by", ['Publish Date', 'Views', 'Engagement Rate', 'Duration (Mins)'])

            with filter_cols[1]:
                sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True)

            with filter_cols[2]:
                num_videos = st.slider("Show videos", 10, min(100, len(df)), min(25, len(df)))

            # Display table
            ascending = (sort_order == 'Ascending')
            display_df = df.sort_values(sort_by, ascending=ascending).head(num_videos)

            st.dataframe(
                display_df[['Title', 'Views', 'Engagement Rate', 'Duration (Mins)', 'Publish Date', 'URL']], 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("Watch", display_text="▶️"),
                    "Views": st.column_config.NumberColumn("Views", format="%d"),
                    "Engagement Rate": st.column_config.NumberColumn("Engagement", format="%.2f%%"),
                    "Duration (Mins)": st.column_config.NumberColumn("Duration", format="%.1f min"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Publish Date": st.column_config.DatetimeColumn("Published", format="MMM D, YYYY")
                },
                height=500
            )

            st.markdown("---")

            # Export options
            st.markdown("### 📥 Export Data")

            export_cols = st.columns(3)

            with export_cols[0]:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV",
                    csv_data,
                    file_name=f"{data['name']}_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with export_cols[1]:
                export_json = {
                    "channel": data['name'],
                    "metrics": {
                        "growth": float(growth_rate),
                        "consistency": float(consistency_score),
                        "diversity": float(diversity_score)
                    },
                    "videos": df.to_dict('records')
                }

                st.download_button(
                    "📄 Download JSON",
                    json.dumps(export_json, indent=2, default=str),
                    file_name=f"{data['name']}_data.json",
                    mime="application/json",
                    use_container_width=True
                )

            with export_cols[2]:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.cache_data.clear()
                    st.success("✅ Cache cleared!")

    else:
        st.warning("⚠️ No video data available.")
        if st.button("🔄 Try Another Search", type="primary"):
            st.session_state['report_data'] = None
            st.rerun()

# Footer
st.markdown("---")
footer_cols = st.columns(3)
footer_cols[0].caption("🚀 Vantio AI v4.1")
footer_cols[1].caption("Powered by Claude AI & Groq")
footer_cols[2].caption(f"© {datetime.now().year}")

print(">>> VANTIO v4.1 LOADED SUCCESSFULLY <<<")