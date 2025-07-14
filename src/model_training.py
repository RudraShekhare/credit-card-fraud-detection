from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_models(X_train, y_train):
    models = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    xgb = XGBClassifier(eval_metric='logloss')  # removed 'use_label_encoder'
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    return models

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/creditcard.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    models = train_models(X_train, y_train)

    # Save model and scaler
    joblib.dump(models['Random Forest'], 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
