import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

#data preparation
# Load the dataset and add exception handling
try:
    df = pd.read_csv("ABV_LOS_Flight357_359_401.csv")
except FileNotFoundError:
    print("Error: The file 'ABV_LOS_Flight357_359_401' was not found.")
    exit()

if df.empty:
    raise ValueError("The dataset is empty. Please check the file path.")
print('\nLoading dataset...')

print("Dataset loaded successfully. Here are a  few random  rows:")
# print(df.head(10))
df['flight_date'] = pd.to_datetime(df['flight_date'])

# print("A random sample of 10 flights details with flight dates.
print(df[df['flight_date'].dt.year != 2025].sample(10))

df["flight_date"] = pd.to_datetime(df["flight_date"])

# adding churn labels and days_since_last_flight
df = df.sort_values(by=["name", "flight_date"])
last_flights = df.groupby("name")["flight_date"].max().reset_index()
last_flights["days_since_last_flight"] = (datetime.now() - last_flights["flight_date"]).dt.days
churn_threshold = 180
last_flights["churn"] = (last_flights["days_since_last_flight"] > churn_threshold).astype(int)

df = df.merge(last_flights[["name", "churn", "days_since_last_flight"]], on="name", how="left")

# --- feature engineering ---

df["flight_month"] = df["flight_date"].dt.month
df["day_of_week"] = df["flight_date"].dt.dayofweek

encoders = {}
for col in ["from", "to", "route", "class"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

features = ["from", "to", "route", "support_calls", "delay_minutes", "class", "flight_month", "day_of_week"]
X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ---model training and prediction ---

model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

y_proba_test = model.predict_proba(X_test)[:, 1]
PREDICTION_THRESHOLD = 0.70
y_pred_test = (y_proba_test >= PREDICTION_THRESHOLD).astype(int)


# ---evaluate and visualization---

print("\n--- Model Evaluation (on Test Set) ---")
print(classification_report(y_test, y_pred_test, zero_division=0))

# ... (rest of the visualization code) ...


# ---prediction on full dataset and identify custormers at risk of churn--- 

print("\n--- Predicting on Full Dataset ---")
full_data_proba = model.predict_proba(X)[:, 1]
df['churn_probability'] = full_data_proba
df['predicted_churn'] = (df['churn_probability'] >= PREDICTION_THRESHOLD).astype(int)

potential_churners = df[df['predicted_churn'] == 1].copy()
potential_churners = potential_churners.sort_values('flight_date', ascending=False).drop_duplicates('name')

potential_churners = potential_churners[['name', 'PNR', 'churn_probability', 'days_since_last_flight']]

print(f"\nFound {len(potential_churners)} passengers likely to churn (Confidence > {int(PREDICTION_THRESHOLD*100)}%):")
if not potential_churners.empty:
    print(potential_churners.to_string(index=False))
else:
    print("No passengers met the churn threshold.")


# --- saving the result outputs ---

output_filename = "flight_churn_predictions_with_scores.csv"
df.to_csv(output_filename, index=False)
print(f"\nFull prediction results saved to '{output_filename}'")