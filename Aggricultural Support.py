# This example predicts the amount of rainfall, a key parameter for agriculture
target = 'Rainfall'  # Column for rainfall predictions

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest for regression)
rainfall_model = RandomForestRegressor(n_estimators=100, random_state=42)
rainfall_model.fit(X_train, y_train)

# Predict rainfall for future periods
rainfall_pred = rainfall_model.predict(X_test)

# Plot predicted vs actual rainfall
plt.plot(rainfall_pred, label="Predicted Rainfall")
plt.plot(y_test.values, label="Actual Rainfall")
plt.legend()
plt.show()
