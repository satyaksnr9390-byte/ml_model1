import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
df = pd.read_csv("energy_level.csv")

X = df[["sleep_hours","break_time"]]
y = df["energy_level"]  

model = LinearRegression()
model.fit(X, y)



BASE = os.path.dirname(__file__)
csv_path = os.path.join(BASE, "energy_level.csv")

@st.cache_data
def load_model():
    df = pd.read_csv(csv_path)
    X = df[["sleep_hours", "break_time"]]
    y = df["energy_level"]
    model = LinearRegression()
    model.fit(X, y)
    return model, df

model, df = load_model()

st.title("Energy Level Predictor")
st.markdown("Input values with sliders and choose algorithm from dropdown.")

sleep_hours = st.slider("Sleep hours", min_value=0.0, max_value=12.0, step=0.25, value=7.0)
break_time = st.slider("Break time", min_value=0.0, max_value=120.0, step=5.0, value=30.0)

algo = st.selectbox("Algorithm", ["LinearRegression"], index=0)
st.write(f"Chosen: {algo}")

if st.button("Predict"):
    pred = model.predict([[sleep_hours, break_time]])
    st.success(f"Predicted energy level: {pred[0]:.3f}")

st.write("---")
st.write("Data preview:")
st.dataframe(df.head())
