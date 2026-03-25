import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Dummy dataset (replace later if needed)
data = pd.DataFrame({
    "team": ["MI","CSK","RCB","KKR","MI","CSK","RCB","KKR"],
    "opponent": ["CSK","MI","KKR","RCB","KKR","RCB","MI","CSK"],
    "powerplay": [50,45,60,48,55,52,65,49],
    "runs_last_5": [45,50,60,48,55,52,65,49],
    "wickets": [2,3,2,4,1,2,2,3],
    "pitch": ["bat","bowl","bat","bowl","bat","bat","bat","bowl"],
    "toss": ["win","lose","win","lose","win","lose","win","lose"],
    "score": [180,175,190,165,185,178,200,170]
})

# Encode categorical
le = LabelEncoder()
for col in ["team","opponent","pitch","toss"]:
    data[col] = le.fit_transform(data[col])

X = data.drop("score", axis=1)
y = data["score"]

# Deep Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=50)

model.save("model.h5")
