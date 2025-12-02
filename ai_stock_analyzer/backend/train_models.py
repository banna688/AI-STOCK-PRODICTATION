# Script to train demo models for a given ticker.
# Usage: python train_models.py TICKER
import sys
import yfinance as yf
from ml_model import TrendModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', help='Ticker to use for training (e.g. AAPL)')
    args = parser.parse_args()
    df = yf.download(args.ticker, period='2y', interval='1d', progress=False)
    if df.empty:
        print('Could not download data. Check internet or ticker.')
        return
    model = TrendModel()
    res = model.train_demo(df)
    print('Training completed. Accuracy on test set:', res['accuracy'])

if __name__ == '__main__':
    main()
