# Fan Sentiment Risk Analysis and API

NLP and machine learning project for estimating airline-level churn risk from social media sentiment. The repository combines transformer-based sentiment analysis, feature engineering, model evaluation, experiment tracking, and a FastAPI service for inference.

## Overview

The project analyzes 14,640 airline tweets and translates sentiment signals into airline-level risk indicators. It is designed as a portfolio project that demonstrates practical NLP, model evaluation, and lightweight API delivery.

## Workflow

```text
Airline tweets
  -> RoBERTa sentiment classification
  -> tweet-level and airline-level feature engineering
  -> churn-risk scoring
  -> model evaluation and SHAP analysis
  -> FastAPI service for inference
```

## Results

### Exploratory Data Analysis
![EDA](data/eda_plots.png)

### Risk Dashboard
![Risk Dashboard](data/risk_dashboard.png)

### Airline Risk Scores
![Risk Scores](data/risk_scores_final.png)

### SHAP Feature Importance
![SHAP](data/shap_final.png)

### Model Evaluation
![Confusion Matrix](data/confusion_matrix.png)

## Key Findings

- RoBERTa provides stronger domain fit than simple lexicon-based sentiment methods for airline tweets.
- Airline-level sentiment patterns are sufficiently distinct to support useful risk ranking.
- Complaint intensity clusters during predictable time windows, which can inform support staffing.
- The repo demonstrates a full loop from experimentation to API exposure.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | API metadata |
| GET | `/health` | Service health check |
| POST | `/predict` | Single tweet sentiment analysis |
| POST | `/predict/batch` | Batch sentiment analysis |
| GET | `/risk-scores` | Airline-level risk scores |
| GET | `/risk-scores/{airline}` | Risk details for one airline |

## Example Response

```json
{
  "text": "US Airways lost my baggage again.",
  "sentiment": "negative",
  "confidence": 0.9557,
  "risk_contribution": "High risk contribution from negative sentiment"
}
```

## Repository Structure

```text
fan-churn-prediction/
|-- data/
|   |-- Tweets.csv
|   |-- tweets_with_sentiment.csv
|   |-- airline_risk_scores_final.csv
|   |-- eda_plots.png
|   |-- risk_dashboard.png
|   |-- risk_scores_final.png
|   |-- shap_final.png
|   `-- confusion_matrix.png
|-- notebooks/
|   `-- 01_EDA_and_Sentiment.ipynb
|-- src/
|   `-- app.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
git clone https://github.com/pavankalmanoor/fan-churn-prediction.git
cd fan-churn-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn src.app:app --reload
```

Then open `http://localhost:8000/docs` for the generated API documentation.

## Technical Stack

| Layer | Tools |
| --- | --- |
| NLP model | Hugging Face Transformers (RoBERTa) |
| Machine learning | XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

## Notes

This project is best interpreted as a portfolio-quality NLP system rather than a production deployment. A real deployment would require stricter monitoring, rate limiting, model artifact management, and a reproducible serving pipeline.
