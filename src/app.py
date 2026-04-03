from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RISK_SCORES_PATH = PROJECT_ROOT / "data" / "airline_risk_scores_final.csv"

app = FastAPI(
    title="Fan Sentiment Risk API",
    description="Analyze airline tweet sentiment and return risk-oriented summaries.",
    version="1.1.0",
)

print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1,
)
print("Sentiment model ready.")

risk_df = pd.read_csv(RISK_SCORES_PATH)
risk_df["airline_normalized"] = risk_df["airline"].str.lower()
risk_df["risk_tier"] = risk_df["risk_tier"].str.replace("[^A-Za-z ]", "", regex=True).str.strip()


class TweetRequest(BaseModel):
    text: str = Field(..., min_length=1)
    airline: Optional[str] = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    airline: Optional[str] = None


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    risk_contribution: str


def classify_risk_contribution(sentiment: str, confidence: float) -> str:
    if sentiment == "negative" and confidence > 0.8:
        return "High risk contribution from negative sentiment"
    if sentiment == "negative":
        return "Moderate risk contribution from negative sentiment"
    if sentiment == "neutral":
        return "Neutral sentiment with limited immediate risk contribution"
    return "Low risk contribution from positive sentiment"


@app.get("/")
def root() -> dict:
    return {
        "message": "Fan Sentiment Risk API",
        "endpoints": ["/predict", "/predict/batch", "/risk-scores", "/health"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TweetRequest) -> SentimentResponse:
    result = sentiment_pipeline(request.text, truncation=True, max_length=512)[0]
    sentiment = result["label"]
    confidence = round(result["score"], 4)

    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        risk_contribution=classify_risk_contribution(sentiment, confidence),
    )


@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> dict:
    results = sentiment_pipeline(request.texts, truncation=True, max_length=512, batch_size=16)

    return {
        "total": len(request.texts),
        "predictions": [
            {
                "text": text[:100],
                "sentiment": result["label"],
                "confidence": round(result["score"], 4),
            }
            for text, result in zip(request.texts, results)
        ],
        "summary": {
            "negative": sum(1 for result in results if result["label"] == "negative"),
            "neutral": sum(1 for result in results if result["label"] == "neutral"),
            "positive": sum(1 for result in results if result["label"] == "positive"),
        },
    }


@app.get("/risk-scores")
def get_risk_scores() -> dict:
    columns = ["airline", "risk_score", "risk_tier", "pct_negative", "pct_positive", "total_tweets"]
    return {
        "airlines": risk_df[columns].to_dict(orient="records"),
        "highest_risk": risk_df.iloc[0]["airline"],
        "lowest_risk": risk_df.iloc[-1]["airline"],
    }


@app.get("/risk-scores/{airline}")
def get_airline_risk(airline: str) -> dict:
    row = risk_df[risk_df["airline_normalized"] == airline.lower()]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Airline '{airline}' not found", "available": risk_df["airline"].tolist()},
        )
    return row.drop(columns=["airline_normalized"]).to_dict(orient="records")[0]
