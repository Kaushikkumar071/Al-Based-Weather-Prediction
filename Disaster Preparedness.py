# Predict the likelihood of a disaster event (like a flood or heatwave)
target = 'DisasterRisk'  # Assuming a binary classification (0: No risk, 1: Risk)

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model
disaster_model = RandomForestClassifier(n_estimators=100, random_state=42)
disaster_model.fit(X_train, y_train)

# Predict disaster risk
disaster_pred = disaster_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, disaster_pred))
print(f'Accuracy: {accuracy_score(y_test, disaster_pred)}')
