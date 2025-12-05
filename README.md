# Wikipedia-Sentiment–Driven Bitcoin Price Forecasting
Alternative data–based ML pipeline for modeling behavioral influence on crypto price dynamics.

This project develops an ML-driven forecasting framework that uses Wikipedia edit logs and NLP-derived sentiment to model short-term BTC/USDT price movements. It evaluates whether collective human-activity signals carry measurable predictive structure when aligned with market data.

---

## 1. Motivation
Financial markets are noisy and difficult to predict using price-only features. This project investigates whether non-financial behavioral data—specifically edit activity and sentiment on the Bitcoin Wikipedia page—can improve short-term directional forecasting.

---

## 2. Data Sources

### Market Data
- BTC/USDT OHLCV from Yahoo Finance (`yfinance`)

### Alternative Data
- Wikipedia edit history of the *Bitcoin* page (via `mwclient`)
- Sentiment extracted from edit comments using a Transformer-based sentiment classifier (HuggingFace)

All signals are merged and time-aligned at daily resolution.

---

## 3. Methodology

### Feature Engineering
- Daily edit counts  
- Daily mean sentiment scores  
- Lagged and rolling-window features  
- Combined behavioral + price-derived predictors

### Model
A regression/classification pipeline predicting next-day BTC direction using engineered features.

### Backtest
A simple long-only strategy:
- Go long when the model predicts an upward move  
- Stay out otherwise  

Performance is compared against a buy-and-hold baseline.

---

## 4. Results
- Wikipedia-derived features exhibit weak but consistent predictive structure.  
- Sentiment adds incremental signal beyond edit frequency.  
- The ML-based strategy shows marginal directional improvement over buy-and-hold on the test window.

(Refer to the notebook for plots and detailed results.)

---

## 5. How to Run

Install dependencies:
```bash
pip install mwclient transformers yfinance tqdm pandas scikit-learn matplotlib
jupyter notebook
```

## 6. Possible Extensions
- Add technical indicators (moving averages, RSI, volatility)
- Use a finance-specific sentiment model
- Extend to multi-step forecasting
- Try sequence models (LSTM, TCN, Transformers)
- Incorporate transaction costs and realistic trading constraints
