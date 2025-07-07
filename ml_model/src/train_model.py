import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import  cross_val_score
from preprocess import load_and_preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train on full dataset for cross-validation
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Initialize Random Forest classifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print("\nCross-Validation F1 Scores:", cv_scores)
print("Average F1 Score:", cv_scores.mean())

# Fit model to training data (only)
model.fit(X_train, y_train)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# Predict on test data
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['high', 'medium', 'low']))

# Save trained model
joblib.dump(model, 'ml_model/models/rf.pkl')
print("\n✅ Model trained and saved successfully.")
