import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and prepare data
df = pd.read_csv("cars.csv").dropna()
X = pd.get_dummies(df.drop("Price", axis=1))
y = df["Price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "car_price_model.pkl")
