import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title='AI Stock Analyzer (Demo)', layout='wide')

st.title('AI Stock Analyzer — Demo')
st.markdown('This is a lightweight demo. Start the backend (FastAPI) then use this Streamlit app.')

ticker = st.text_input('Ticker (e.g. AAPL, MSFT, TCS.NS)', value='AAPL')

col1, col2 = st.columns(2)
with col1:
    if st.button('Fetch & Show Price Chart'):
        df = yf.download(ticker, period='6mo', interval='1d', progress=False)
        if df.empty:
            st.error('Could not fetch data for ticker. Check internet and ticker symbol.')
        else:
            st.subheader(f'Price chart for {ticker}')
            st.line_chart(df['Close'])
            st.dataframe(df.tail(5))

with col2:
    if st.button('Call Backend Predictions'):
        try:
            resp = requests.post('http://localhost:8000/predict_trend', json={'ticker': ticker}, timeout=15).json()
            resp2 = requests.post('http://localhost:8000/predict_next_price', json={'ticker': ticker}, timeout=15).json()
            st.subheader('Trend Prediction')
            st.json(resp)
            st.subheader('Next Price Estimate')
            st.json(resp2)
        except Exception as e:
            st.error('Could not reach backend. Make sure backend is running on http://localhost:8000')
            st.exception(e)

st.markdown('---')
st.markdown('**How to run (quick)**')
st.code('''
1) Install requirements: pip install -r requirements.txt
2) Start backend: cd backend && uvicorn app:app --reload
3) (optional) Train demo models: python train_models.py AAPL
4) Start frontend: streamlit run frontend/app.py
''')
