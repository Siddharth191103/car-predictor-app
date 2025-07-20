import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and model
df = pd.read_csv("cars.csv").dropna()
model = joblib.load("car_price_model.pkl")

# Prepare training features for reference
X_train = pd.get_dummies(df.drop("Price", axis=1))

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")

st.sidebar.header("Enter Car Features")

# Build the user input form
def user_input_features():
    input_data = {}
    for col in df.drop("Price", axis=1).columns:
        if df[col].dtype == 'object':
            input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:
            input_data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    return pd.DataFrame([input_data])

# Capture user input
input_df = user_input_features()

# One-hot encode user input
input_encoded = pd.get_dummies(input_df)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict
prediction = model.predict(input_encoded)[0]

# Output
st.subheader("📈 Prediction Result")
st.success(f"Estimated Price: ₹ {prediction:,.2f}")

st.subheader("🧾 Input Features")
st.write(input_df)

st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

