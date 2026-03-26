
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv("energy_level.csv")

X = df[["sleep_hours","break_time"]]
y = df["energy_level"]  

model = LinearRegression()
model.fit(X, y)

sleep_hours = float(input("Enter sleep hours: "))
break_time = float(input("Enter break time: "))


prediction = model.predict([[sleep_hours, break_time]])
print(f"Predicted energy level: {prediction[0]}")     
