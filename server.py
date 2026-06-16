import os
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

# Try loading transformers (RoBERTa)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = None
model = None
roberta_available = False
roberta_error = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    import torch
    print("Loading CardiffNLP RoBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    roberta_available = True
    print("RoBERTa model loaded successfully!")
except Exception as e:
    roberta_error = str(e)
    print(f"Warning: CardiffNLP RoBERTa model not loaded. Error: {e}")

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
    if roberta_available:
        try:
            encoded_text = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            roberta_scores = {
                "neg": float(scores[0]),
                "neu": float(scores[1]),
                "pos": float(scores[2])
            }
        except Exception as e:
            print(f"RoBERTa analysis failed: {e}")
            
    return {
        "text": text,
        "vader": vader_scores,
        "roberta": roberta_scores,
        "roberta_available": roberta_available and roberta_scores is not None
    }

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
