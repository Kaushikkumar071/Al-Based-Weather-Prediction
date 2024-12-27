# For actionable insights, you could analyze the data and generate forecasts
future_data = pd.read_csv('future_weather_data.csv')

# Make predictions for future weather
future_predictions = model.predict(future_data[features])

# Provide actionable insights
for i, prediction in enumerate(future_predictions):
    if prediction > 30:  # Threshold for heat warning
        print(f"Warning: High temperature predicted on day {i+1}: {prediction}°C")
    elif prediction < 0:  # Threshold for frost warning
        print(f"Warning: Frost predicted on day {i+1}")
    else:
        print(f"Day {i+1} - Safe weather with a temperature of {prediction}°C")
