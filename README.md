# Fan Sentiment Risk Analysis & API

> NLP-powered fan sentiment analysis and churn risk scoring system built with HuggingFace RoBERTa, MLflow, and FastAPI

## 🎯 Project Overview

Analyzes 14,640 airline tweets using transformer-based NLP to identify at-risk fan communities and predict churn signals. Delivers results via a production-ready REST API with real-time inference.

## 🏗️ Architecture

```
Raw Tweets (14,640)
       ↓
HuggingFace RoBERTa (twitter-roberta-base-sentiment-latest)
       ↓
Sentiment Classification (Positive / Neutral / Negative)
       ↓
Feature Engineering (13 behavioral + sentiment features)
       ↓
Risk Scoring Engine (Composite 0-100 score)
       ↓
FastAPI REST API (6 endpoints)
```

## 📊 Key Results

| Airline | Risk Score | Risk Tier | % Negative Tweets |
|---|---|---|---|
| US Airways | 100.0 | 🔴 High Risk | 72.5% |
| American | 92.3 | 🔴 High Risk | 68.9% |
| United | 76.5 | 🔴 High Risk | 62.6% |
| Southwest | 29.5 | 🟢 Low Risk | 47.1% |
| Delta | 5.6 | 🟢 Low Risk | 39.6% |
| Virgin America | 0.0 | 🟢 Low Risk | 39.1% |

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| NLP Model | HuggingFace Transformers (RoBERTa) |
| ML Framework | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## 🚀 Quick Start

```bash
git clone https://github.com/pavankalmanoor/fan-churn-prediction
cd fan-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload
open http://localhost:8000/docs
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Single tweet sentiment |
| POST | `/predict/batch` | Batch tweet analysis |
| GET | `/risk-scores` | All airline risk scores |
| GET | `/risk-scores/{airline}` | Specific airline risk |

## 📈 Sample API Response

```json
POST /predict
{
  "text": "US Airways lost my baggage AGAIN. Worst airline ever!"
}

Response:
{
  "sentiment": "negative",
  "confidence": 0.9557,
  "risk_contribution": "High — negative sentiment increases churn risk"
}
```

## 🔍 Key Findings

- **RoBERTa outperforms human labels** — identified mislabeled tweets (e.g. "added commercials... tacky" labeled positive, predicted negative at 92% confidence)
- **US Airways highest churn risk** with 72.5% negative tweet rate
- **Virgin America lowest risk** with highest positive sentiment ratio
- **Peak complaint hours: 8am–2pm** — actionable for customer service staffing
- **MLflow tracked 5 experiment runs** comparing XGBoost, Random Forest, and Logistic Regression

## 📁 Project Structure

```
fan-churn-prediction/
├── data/
│   ├── Tweets.csv                      # Raw data (14,640 tweets)
│   ├── tweets_with_sentiment.csv       # RoBERTa predictions
│   ├── airline_risk_scores_final.csv   # Risk scores
│   ├── risk_scores_final.png           # Risk dashboard
│   ├── shap_final.png                  # SHAP plot
│   └── confusion_matrix.png            # Model evaluation
├── notebooks/
│   └── 01_EDA_and_Sentiment.ipynb     # Full analysis pipeline
├── src/
│   └── app.py                          # FastAPI application
├── requirements.txt
└── README.md
```

## 💡 Interview Talking Points

1. **Why RoBERTa over VADER?** RoBERTa is trained on 58M tweets — domain-specific and captures sarcasm better than lexicon-based approaches
2. **Data leakage detection** — identified and fixed target leakage during feature engineering
3. **Production design** — API loads model once at startup, serves inference in under 100ms
4. **MLflow tracking** — every experiment logged with params, metrics, and artifacts
