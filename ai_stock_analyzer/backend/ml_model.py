# Simple lightweight ML model helpers for demo purposes.
# Uses scikit-learn RandomForest for trend classification and regression.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'trend_clf.joblib')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'price_reg.joblib')

class TrendModel:
    def __init__(self):
        # Load models if exist, else create new simple ones (untrained)
        self.clf = None
        self.reg = None
        if os.path.exists(CLASSIFIER_PATH):
            self.clf = joblib.load(CLASSIFIER_PATH)
        if os.path.exists(REGRESSOR_PATH):
            self.reg = joblib.load(REGRESSOR_PATH)

    def prepare_features(self, df):
        # Expect df indexed by date with Open, High, Low, Close, Volume
        df = df.copy()
        df['return_1'] = df['Close'].pct_change()
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_10'] = df['Close'].rolling(10).mean()
        df['rsi_like'] = (df['Close'] - df['Close'].rolling(14).min()) / (df['Close'].rolling(14).max() - df['Close'].rolling(14).min() + 1e-9)
        df = df.dropna()
        features = df[['return_1','ma_5','ma_10','rsi_like','Volume']].tail(1)
        return features

    def train_demo(self, df):
        # Train simple models on historic data (super quick)
        df = df.copy().dropna()
        df['return_1'] = df['Close'].pct_change()
        df['target_trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 if next day up
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_10'] = df['Close'].rolling(10).mean()
        df['rsi_like'] = (df['Close'] - df['Close'].rolling(14).min()) / (df['Close'].rolling(14).max() - df['Close'].rolling(14).min() + 1e-9)
        df = df.dropna()
        X = df[['return_1','ma_5','ma_10','rsi_like','Volume']]
        y = df['target_trend']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        acc = accuracy_score(y_test, ypred)
        # Train regressor for next day close
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(X_train, df.loc[X_train.index, 'Close'].shift(-1).fillna(method='ffill'))
        # Save
        joblib.dump(clf, CLASSIFIER_PATH)
        joblib.dump(reg, REGRESSOR_PATH)
        self.clf = clf
        self.reg = reg
        return {'accuracy': float(acc)}

    def predict_trend(self, features_df):
        if self.clf is None:
            # Not trained: fall back to simple heuristic
            row = features_df.iloc[-1]
            pred = pred = 'UP' if row['ma_5'].iloc[-1] > row['ma_10'].iloc[-1] else 'DOWN'

            conf = 0.6
            return pred, conf
        p = self.clf.predict_proba(features_df)[:,1]
        prob = p[-1]
        return ('UP' if prob>=0.5 else 'DOWN'), float(prob)

    def predict_next_price(self, df):
        # Predict next close price using regressor if available else naive next = last close * (1 + return_1)
        last_close = df['Close'].iloc[-1]
        features = self.prepare_features(df)
        if self.reg is None:
            last_ret = features['return_1'].iloc[-1]
            return last_close * (1 + last_ret)
        pred = self.reg.predict(features)
        return float(pred[0])
