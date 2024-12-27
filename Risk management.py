from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming a column 'Event' that classifies weather as 'Normal', 'Extreme', 'Severe'
X = data[features]
y = data['Event']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
