import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================
# 1️⃣ Load Dataset
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "nassau_dataset.csv")

print("Loading dataset...")
data = pd.read_csv(data_path)

print("Available Columns:")
print(data.columns)

# =====================================
# 2️⃣ Data Preprocessing
# =====================================

print("Preprocessing data...")

# Convert dates safely
data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True, errors="coerce")
data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True, errors="coerce")

# Create Lead Time
data["Lead Time"] = (data["Ship Date"] - data["Order Date"]).dt.days

# Remove invalid rows
data = data.dropna(subset=["Lead Time"])
data = data[data["Lead Time"] >= 0]

# =====================================
# 3️⃣ Feature Selection
# =====================================

# Use only columns that exist
possible_features = ["Ship Mode", "Region", "Division"]
features = [col for col in possible_features if col in data.columns]

if len(features) == 0:
    raise Exception("Required feature columns not found in dataset.")

X = data[features].copy()
y = data["Lead Time"]

# =====================================
# 4️⃣ Encode Categorical Variables
# =====================================

print("Encoding categorical variables...")

label_encoders = {}

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# =====================================
# 5️⃣ Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 6️⃣ Train Models
# =====================================

print("Training models...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
gb_model.fit(X_train, y_train)

# =====================================
# 7️⃣ Evaluation Function
# =====================================

def evaluate_model(model, name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

evaluate_model(lr_model, "Linear Regression")
evaluate_model(rf_model, "Random Forest")
evaluate_model(gb_model, "Gradient Boosting")

# =====================================
# 8️⃣ Save Models
# =====================================

print("\nSaving models...")

models_path = os.path.join(BASE_DIR, "models")
os.makedirs(models_path, exist_ok=True)

joblib.dump(rf_model, os.path.join(models_path, "lead_time_model.pkl"))
joblib.dump(rf_model, os.path.join(models_path, "random_forest_model.pkl"))
joblib.dump(gb_model, os.path.join(models_path, "gradient_boosting_model.pkl"))
joblib.dump(label_encoders, os.path.join(models_path, "label_encoders.pkl"))
joblib.dump(features, os.path.join(models_path, "feature_columns.pkl"))

print("\n✅ Model training complete. All models saved successfully.")