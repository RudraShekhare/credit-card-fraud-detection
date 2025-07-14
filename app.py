import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load simulated data
df = pd.read_csv('data/fraud_data_simulated.csv')

st.title("💳 Credit Card Fraud Detection")

# User selects a person or shows fraud case
st.sidebar.header("🔧 Options")
show_fraud = st.sidebar.checkbox("Show me a fraud case", value=False)

if show_fraud:
    fraud_df = df[df['Class'] == 1]
    person = fraud_df.sample(1).iloc[0]
    st.markdown("⚠️ **Showing a randomly selected fraud case**")
else:
    name = st.selectbox("Select Customer Name", df['Name'].unique())
    person = df[df['Name'] == name].iloc[0]

# Display transaction details
st.markdown(f"""
- **Name:** {person['Name']}
- **Card Number:** {person['CardNumber']}
- **Merchant:** {person['Merchant']}
- **City:** {person['City']}
- **Amount:** ₹{person['Amount']}
""")

# Prepare model input
features = person[[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']].values.reshape(1, -1)
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0][1]

st.subheader("🔍 Prediction Result:")
st.write(f"**Fraud Probability:** `{prob:.2%}`")

if prediction == 0:
    st.success("✅ Legitimate Transaction")
else:
    st.error("🚨 Fraudulent Transaction")
