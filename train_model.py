import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------
# 1. Load your dataset
# -------------------------
df = pd.read_csv("construction_data.csv")   # <<< yaha apne CSV ka naam daalo

target = "Total_Estimate"   # <<< yaha apne target column ka naam daalo

df = df.dropna(subset=[target])

# Basic encoding for categoricals
df = pd.get_dummies(df, drop_first=True)

X = df.drop(target, axis=1)
y = df[target]


# -------------------------
# 2. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------
# 3. Linear Regression Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# -------------------------
# 4. Evaluation
# -------------------------
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Model performance:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# -------------------------
# 5. Save model + columns
# -------------------------
model_artifact = {
    "model": model,
    "columns": list(X.columns)
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_artifact, f)

print("\nModel saved as model.pkl")
