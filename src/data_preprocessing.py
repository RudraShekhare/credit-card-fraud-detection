import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    fraud = df[df['Class'] == 1]
    non_fraud = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
    data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))

    X = data.drop('Class', axis=1)
    y = data['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
