from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Fan Sentiment Risk API",
    description="Analyze fan sentiment and predict churn risk using RoBERTa",
    version="1.0.0"
)

# ── Load Model Once at Startup ─────────────────────────
print("⏳ Loading RoBERTa model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1
)
print("✅ Model ready!")

# ── Load Risk Scores ───────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
risk_df = pd.read_csv(PROJECT_ROOT / 'data' / 'airline_risk_scores_final.csv')

# ── Request/Response Models ────────────────────────────
class TweetRequest(BaseModel):
    text: str
    airline: str = None

class BatchRequest(BaseModel):
    texts: list[str]
    airline: str = None

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    risk_contribution: str

# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Fan Sentiment Risk API",
        "endpoints": [
            "/predict — single tweet sentiment",
            "/predict/batch — multiple tweets",
            "/risk-scores — airline risk dashboard",
            "/health — API health check"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "RoBERTa-twitter-sentiment"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TweetRequest):
    result = sentiment_pipeline(
        request.text, truncation=True, max_length=512
    )[0]
    
    sentiment = result['label']
    confidence = round(result['score'], 4)
    
    risk_contribution = (
        "High — negative sentiment increases churn risk" 
        if sentiment == 'negative' and confidence > 0.8
        else "Medium — monitor this fan"
        if sentiment == 'negative'
        else "Low — positive engagement"
    )
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        risk_contribution=risk_contribution
    )

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = sentiment_pipeline(
        request.texts, truncation=True, max_length=512, batch_size=16
    )
    
    return {
        "total": len(request.texts),
        "predictions": [
            {
                "text": text[:100],
                "sentiment": r['label'],
                "confidence": round(r['score'], 4)
            }
            for text, r in zip(request.texts, results)
        ],
        "summary": {
            "negative": sum(1 for r in results if r['label'] == 'negative'),
            "neutral":  sum(1 for r in results if r['label'] == 'neutral'),
            "positive": sum(1 for r in results if r['label'] == 'positive'),
        }
    }

@app.get("/risk-scores")
def get_risk_scores():
    return {
        "airlines": risk_df[['airline','risk_score','risk_tier',
                              'pct_negative','pct_positive',
                              'total_tweets']].to_dict(orient='records'),
        "highest_risk": risk_df.iloc[0]['airline'],
        "lowest_risk":  risk_df.iloc[-1]['airline']
    }

@app.get("/risk-scores/{airline}")
def get_airline_risk(airline: str):
    row = risk_df[risk_df['airline'].str.lower() == airline.lower()]
    if row.empty:
        return {"error": f"Airline '{airline}' not found",
                "available": risk_df['airline'].tolist()}
    return row.to_dict(orient='records')[0]