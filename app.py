import streamlit as st
import pandas as pd
import joblib
from faker import Faker

# Load model and scaler
model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load reduced dataset
df = pd.read_csv('data/fraud_data_simulated.csv')

# 🔁 Dynamically add fake metadata
fake = Faker()
df['Name'] = [fake.name() for _ in range(len(df))]
df['CardNumber'] = [fake.credit_card_number() for _ in range(len(df))]
df['Merchant'] = [fake.company() for _ in range(len(df))]
df['City'] = [fake.city() for _ in range(len(df))]

# App title
st.title("💳 Credit Card Fraud Detection")

# Sidebar options
st.sidebar.header("🔧 Options")
show_fraud = st.sidebar.checkbox("Show me a fraud case", value=False)

# Choose transaction
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

# Prepare features for model
features = person[[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']].values.reshape(1, -1)
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0][1]

st.subheader("🔍 Prediction Result:")

if prediction == 0:
    st.success("✅ Legitimate Transaction")
else:
    st.error("🚨 Fraudulent Transaction")

st.markdown(f"**Fraud Probability:** `{prob:.2%}`")
