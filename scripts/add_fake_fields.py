import pandas as pd
import random
from faker import Faker

df = pd.read_csv("data/creditcard.csv")
fake = Faker()

df['Name'] = [fake.name() for _ in range(len(df))]
df['CardNumber'] = [''.join(random.choices('0123456789', k=16)) for _ in range(len(df))]
df['Merchant'] = random.choices(['Amazon', 'Flipkart', 'Netflix', 'Apple'], k=len(df))
df['City'] = random.choices(['Mumbai', 'Delhi', 'New York', 'London'], k=len(df))

df.to_csv("data/fraud_data_simulated.csv", index=False)
