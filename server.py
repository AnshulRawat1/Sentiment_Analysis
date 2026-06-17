import os
import socket

# Force IPv4 DNS resolution to prevent "[Errno -5] No address associated with hostname" 
# on IPv4-only host networks (like Render containers) which fail on IPv6 (AAAA) resolution.
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(*args, **kwargs):
    if 'family' in kwargs:
        if kwargs['family'] == 0:
            kwargs['family'] = socket.AF_INET
    elif len(args) >= 3:
        args = list(args)
        if args[2] == 0:
            args[2] = socket.AF_INET
    else:
        args = list(args)
        while len(args) < 3:
            args.append(0)
        args[2] = socket.AF_INET
    return _orig_getaddrinfo(*args, **kwargs)

socket.getaddrinfo = _ipv4_only_getaddrinfo

import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Download VADER lexicon
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"Warning: NLTK downloader failed: {e}")

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Hugging Face Inference API configuration
import json
import time
import urllib.request
import urllib.error

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")

roberta_available = True
roberta_error = None

if not HF_API_TOKEN:
    print("Warning: Neither HF_API_TOKEN nor HF_TOKEN environment variable is set. Hugging Face Inference API calls may fail or be rate-limited.")

def query_hf_api(text: str, max_retries=3, delay=5):
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                HF_API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                # Check for model loading error
                if isinstance(result, dict) and "error" in result and "currently loading" in result["error"]:
                    est_time = result.get("estimated_time", delay)
                    print(f"Model is loading, waiting {est_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(min(est_time, 10))  # sleep up to 10s
                    continue
                return result
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                if isinstance(err_data, dict) and "error" in err_data and "currently loading" in err_data["error"]:
                    est_time = err_data.get("estimated_time", delay)
                    print(f"Model is loading (HTTP {e.code}), waiting {est_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(min(est_time, 10))
                    continue
                raise Exception(err_data.get("error", str(e)))
            except Exception:
                raise Exception(f"HTTP Error {e.code}: {e.reason}")
        except Exception as e:
            raise e
            
    raise Exception("Model is still loading. Please try again in a few moments.")

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/api/status")
def get_status():
    return {
        "vader_status": "loaded",
        "roberta_status": "loaded" if roberta_available else "failed/unavailable",
        "roberta_error": roberta_error
    }

@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # 1. NLTK VADER sentiment analysis
    try:
        vader_scores = sia.polarity_scores(text)
    except Exception as e:
        vader_scores = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0, "error": str(e)}
        
    # 2. RoBERTa sentiment analysis
    roberta_scores = None
    roberta_err_msg = None
    if roberta_available:
        try:
            result = query_hf_api(text)
            
            # Map the API response format back to neg, neu, pos
            scores_dict = {"neg": 0.0, "neu": 0.0, "pos": 0.0}
            
            if isinstance(result, list) and len(result) > 0:
                first_el = result[0]
                items = first_el if isinstance(first_el, list) else result
                
                for item in items:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        label = str(item["label"]).lower()
                        score = float(item["score"])
                        if label in ["label_0", "negative", "neg"]:
                            scores_dict["neg"] = score
                        elif label in ["label_1", "neutral", "neu"]:
                            scores_dict["neu"] = score
                        elif label in ["label_2", "positive", "pos"]:
                            scores_dict["pos"] = score
                            
                roberta_scores = scores_dict
        except Exception as e:
            roberta_err_msg = str(e)
            print(f"RoBERTa analysis failed: {e}")
            global roberta_error
            roberta_error = str(e)
            
    return {
        "text": text,
        "vader": vader_scores,
        "roberta": roberta_scores,
        "roberta_available": roberta_available and roberta_scores is not None,
        "roberta_error": roberta_err_msg
    }

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
