# AI Stock Analyzer — Demo (Ready-to-run project folder)

This is a lightweight demo of the AI Stock Analyzer project we discussed.
It intentionally keeps models simple so you can run and test locally quickly.

## What's included
- `backend/` — FastAPI service with endpoints:
    - `/predict_trend` — returns UP/DOWN with confidence
    - `/predict_next_price` — returns a next-close price estimate
  Also includes `train_models.py` to train demo models using yfinance data.

- `frontend/` — Streamlit app that fetches price charts and calls backend.

- `requirements.txt` — Python dependencies.

## Quick start (local)
1. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. Start backend
   ```
   cd backend
   uvicorn app:app --reload
   ```

3. (Optional) Train demo models for a ticker
   ```
   python train_models.py AAPL
   ```

4. Open the frontend in another terminal
   ```
   streamlit run frontend/app.py
   ```

## Notes
- The demo uses yfinance to fetch historical prices. Internet required for live data.
- Models saved under `backend/models/` after training.
- This is a starting point — you can swap the TrendModel with LSTM/XGBoost/FinBERT later.

## Need full production-grade version?
If you want, I can now:
- Replace trend model with XGBoost + feature engineering
- Add FinBERT-based sentiment and news ingestion
- Add authentication, PostgreSQL, Dockerfile, and Kubernetes manifests
- Build a React dashboard instead of Streamlit

Tell me which upgrades you want next.
