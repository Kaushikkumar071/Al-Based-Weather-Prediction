import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load weather data (assume it's a CSV file with temperature and other features)
data = pd.read_csv('weather_data.csv')

# Preprocess data (handling missing values, feature engineering, etc.)
data = data.fillna(method='ffill')

# Select relevant features
features = ['Humidity', 'WindSpeed', 'Pressure']
target = 'Temperature'

# Create X (features) and y (target)
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
