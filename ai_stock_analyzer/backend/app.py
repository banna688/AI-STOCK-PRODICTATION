from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from ml_model import TrendModel
import yfinance as yf
import pandas as pd

app = FastAPI(title='AI Stock Analyzer - Backend (Demo)')

model = TrendModel()

class TickerRequest(BaseModel):
    ticker: str

@app.post('/predict_trend')
async def predict_trend(req: TickerRequest):
    """Fetches last 120 days data, computes simple features, and returns
    a trend prediction (UP / DOWN) with confidence score (demo)."""
    df = yf.download(req.ticker, period='120d', interval='1d', progress=False)
    if df.empty:
        return {'error': 'No data for ticker'}
    df = df.dropna()
    X = model.prepare_features(df)
    pred, conf = model.predict_trend(X)
    return {'ticker': req.ticker, 'prediction': pred, 'confidence': float(conf)}

@app.post('/predict_next_price')
async def predict_next_price(req: TickerRequest):
    df = yf.download(req.ticker, period='120d', interval='1d', progress=False)
    if df.empty:
        return {'error': 'No data for ticker'}
    df = df.dropna()
    next_price = model.predict_next_price(df)
    return {'ticker': req.ticker, 'next_close_estimate': float(next_price)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
