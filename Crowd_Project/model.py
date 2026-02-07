import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("cafeteria_data.csv")

X = data[["Hour", "Day"]]
y = data["Crowd"]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("crowd_model.pkl", "wb"))

print("Model trained successfully!")