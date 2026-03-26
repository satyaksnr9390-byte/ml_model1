import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


csv_path =  "energy_level.csv"

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

sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.25)
break_time = st.slider("Break time", 0.0, 120.0, 30.0, 5.0)

algo = st.selectbox("Algorithm", ["LinearRegression"])
st.write(f"Chosen: {algo}")

if st.button("Predict"):
    input_df = pd.DataFrame([[sleep_hours, break_time]],
                        columns=["sleep_hours", "break_time"])
    pred = model.predict(input_df)
    st.success(f"Predicted energy level: {pred[0]:.3f}")

st.write("---")
st.write("Data preview:")
st.dataframe(df.head())
