# Credit Card Fraud Detection

🎯 **Objective**  
Detect fraudulent credit card transactions using machine learning, featuring an interactive Streamlit web app for demo and exploration.

---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-b26st967l6jrmvbzpumbb6.streamlit.app/)

## 🚀 Features

- Tabular data from Kaggle (284K transactions; 492 fraud)
- Models implemented:
  - **Random Forest** (primary saved model)
  - Logistic Regression & XGBoost (for experimentation)
- Imbalanced dataset handling via scaling & model tuning
- Interactive **Streamlit app** with:
  - Select a simulated user transaction
  - Show fraud probability and decision
  - Option to display a random fraud case
  - Configurable fraud threshold (moving towards)
    
## 📸 App Screenshots

### 🏠 Homepage
![App Screenshot 1](assets/screenshot1.png)

### 🔍 Prediction Output
![App Screenshot 2](assets/screenshot2.png)


---

## 📁 Repo Structure

credit-card-fraud-detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│ ├── creditcard.csv
│ └── fraud_data_simulated.csv
├── models/
│ ├── fraud_model.pkl
│ └── scaler.pkl
├── notebooks/
│ ├── 01_EDA.ipynb
│ └── 02_Model_Training.ipynb
├── scripts/
│ └── add_fake_fields.py
├── src/
│ └── model_training.py
├── app.py


---

## 💻 Setup Instructions

1. **Clone and install dependencies.**

   ```bash
   git clone <YOUR_REPO_URL>
   cd credit-card-fraud-detection
   pip install -r requirements.txt

2. **Download creditcard.csv from [Kaggle dataset].**

Place it in the data/ folder.

Generate simulated data

```bash
python scripts/add_fake_fields.py
Train & save model
```

```bash
python src/model_training.py
Run web app
```
```bash
streamlit run app.py
```

3. **📈 Usage**
 - Use the dropdown to select a "user"
 - Or check “Show me a fraud case” to randomly sample a fraud

  The app shows:

  --> Customer details
  --> Fraud probability
  --> Decision (Legit vs Fraud)


4. **⚙️ Future Improvements**
- Add SMOTE/ADASYN for class imbalance
- Use GridSearchCV or Optuna for hyperparameter tuning
- Add cross-validation & evaluation metrics
- Track experiments with MLflow / Weights & Biases
- Deploy publicly via Streamlit Cloud or Hugging Face Spaces
- Improve UI: add charts, threshold slider, download option


5. **📄 License**
This project is open-source under the MIT License.
See LICENSE for details.


6. **🛠️ Built With**
--> Python, pandas, scikit-learn, XGBoost
--> Streamlit for app UI
--> joblib for model persistence
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-b26st967l6jrmvbzpumbb6.streamlit.app/)


