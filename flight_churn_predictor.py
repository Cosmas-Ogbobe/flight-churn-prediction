import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import time

# Loading the dataset
df = pd.read_csv("ABV_LOS_Flight357_359_401.csv")
# checking if the dataset is loaded correctly
if df.empty:
    raise ValueError("The dataset is empty. Please check the file path and content.")
# Display the first few rows of the dataset
print("\nLoading dataset...")
time.sleep(3)  # Simulate loading time
print('Dataset Overview:')
print("Dataset loaded successfully. Here are the first few rows:")
print(df.head(10))
# Ensure necessary columns are present
required_columns = ["name", "PNR", "flight_date", "from", "to", "route", "class", "support_calls", "delay_minutes"]
missing_columns = [col for col in required_columns if col not in df.columns]        
if missing_columns:
    raise ValueError(f"The dataset is missing the following required columns: {', '.join(missing_columns)}")


# Convert flight_date to datetime
df["flight_date"] = pd.to_datetime(df["flight_date"])

# Add churn labels (threshold between 150-210 days)
df = df.sort_values(by=["name", "flight_date"])
last_flights = df.groupby("name")["flight_date"].max().reset_index()
last_flights["days_since_last_flight"] = (datetime.now() - last_flights["flight_date"]).dt.days
# Random threshold between 150 and 210
threshold = np.random.randint(150, 210)
last_flights["churn"] = last_flights["days_since_last_flight"].apply(lambda x: 1 if x > threshold else 0)
df = df.merge(last_flights[["name", "churn"]], on="name", how="left")

# Introduce label noise (flip 10% of churn labels)
np.random.seed(42)
flip_indices = df.sample(frac=0.1).index
df.loc[flip_indices, "churn"] = 1 - df.loc[flip_indices, "churn"]

# Create new time features
df["flight_month"] = df["flight_date"].dt.month
df["day_of_week"] = df["flight_date"].dt.dayofweek

# Encode categorical features
encoders = {}
for col in ["from", "to", "route", "class"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Add noise to numerical columns
df["support_calls"] = df["support_calls"] + np.random.randint(-1, 2, size=len(df))
df["delay_minutes"] = df["delay_minutes"] + np.random.randint(-10, 11, size=len(df))
df["support_calls"] = df["support_calls"].clip(lower=0)
df["delay_minutes"] = df["delay_minutes"].clip(lower=0)

# Prepare features and target
features = ["from", "to", "route", "support_calls", "delay_minutes", "class", "flight_month", "day_of_week"]
X = df[features]
y = df["churn"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
# model = RandomForestClassifier(random_state=42, class_weight="balanced")
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict on full data
df["predicted_churn"] = model.predict(X)
overall_acc = accuracy_score(df["churn"], df["predicted_churn"])
print(f"\n Overall accuracy on full data: {round(overall_acc * 100, 2)}%")

# Show mismatches
wrong_preds = df[df["churn"] != df["predicted_churn"]]
print(f"\n Wrong predictions: {len(wrong_preds)}")
print(wrong_preds[["name", "PNR", "churn", "predicted_churn"]].head())

# Show who is predicted to churn
churned = df[df["predicted_churn"] == 1][["name", "PNR"]].drop_duplicates()
print("\n Passengers likely to churn:")
print(churned.head(10))

# Save the full output
# df.to_csv("flight_churn_predictions_output_noisy.csv", index=False)
# print("\n Predictions saved to flight_churn_predictions_output_noisy.csv")


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
# plt.savefig("confusion_matrix_flight.png")
plt.show()

plt.figure(figsize=(6, 4))
df['predicted_churn'].value_counts().sort_index().plot(kind='bar', color=['green', 'red'])
plt.xticks([0, 1], ['Not Churned', 'Churned'], rotation=0)
plt.title('Predicted Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Number of Passengers')
plt.tight_layout()
# plt.savefig("predicted_churn_distribution.png")
plt.show()

importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance')

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='coolwarm')
plt.title('Feature Importance - What drives churn?')
plt.tight_layout()
# plt.savefig("feature_importance_flight.png")
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="day_of_week", hue="predicted_churn", palette=["green", "red"])
plt.title("Churn Prediction by Day of Week")
plt.xlabel("Day of Week (0=Mon, 6=Sun)")
plt.ylabel("Passenger Count")
plt.legend(["Not Churned", "Churned"])
plt.tight_layout()
# plt.savefig("churn_by_dayofweek.png")
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="flight_month", hue="predicted_churn", palette=["green", "red"])
plt.title("Churn Prediction by Flight Month")
plt.xlabel("Month")
plt.ylabel("Passenger Count")
plt.legend(["Not Churned", "Churned"])
plt.tight_layout()
# plt.savefig("churn_by_month.png")
plt.show()

