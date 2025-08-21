import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from preprocess import load_and_preprocess_data
import pandas as pd

# ============================
# Ensure output dirs exist
# ============================
os.makedirs("ml_model/results", exist_ok=True)
os.makedirs("ml_model/models", exist_ok=True)

# ============================
# Load preprocessed features + labels
# ============================
X, y, selected_features = load_and_preprocess_data(show_table=True, k_features=10)

print("\nSelected Features for training:")
print(selected_features)

# ============================
# Confusion matrix plotting
# ============================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['High', 'Medium', 'Low']
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')

# ============================
# Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# Handle class imbalance with SMOTE
# ============================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ============================
# Random Forest + Hyperparameter tuning
# ============================
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_resampled, y_train_resampled)
best_model = grid_search.best_estimator_

print("\n✅ Best Hyperparameters:", grid_search.best_params_)

# ============================
# Training / Validation / Test Evaluation
# ============================
train_acc = best_model.score(X_train_resampled, y_train_resampled)
print("Training Accuracy:", round(train_acc * 100, 2), "%")

cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=3, scoring='accuracy')
print("Validation Accuracy (Avg):", round(cv_scores.mean() * 100, 2), "%")

y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(test_acc * 100, 2), "%")

print("\n📊 Classification Report (Multiclass):")
print(classification_report(y_test, y_pred, target_names=['high', 'medium', 'low']))

# ============================
# Save confusion matrix plot
# ============================
plot_confusion_matrix(y_test, y_pred, 'Random Forest')
plt.savefig('ml_model/results/confusion_matrix_rf.png')
plt.close()

# ============================
# Save tuned model
# ============================
joblib.dump(best_model, 'ml_model/models/rf_tuned.pkl')
print("\n✅ Tuned Random Forest model trained and saved successfully.")

# ============================
# Save feature importance
# ============================
importances = best_model.feature_importances_
feat_importances = pd.DataFrame({
    "Feature": selected_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feat_importances.to_csv("ml_model/results/feature_importances.csv", index=False)
print("\n📈 Feature importances saved.")
