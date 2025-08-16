import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from preprocess import load_and_preprocess_data

# Load preprocessed data
X, y = load_and_preprocess_data()
X_scaled, y = load_and_preprocess_data(show_table=True)

# Cross-validation
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print("\nCross-Validation F1 Scores:", cv_scores)
print("Average F1 Score:", cv_scores.mean())

# Final split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE on training set only
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train and evaluate final model
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['high', 'medium', 'low']))

# Save model
joblib.dump(model, 'ml_model/models/rf.pkl')
print("\n✅ Model trained and saved successfully.")


# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
# from imblearn.over_sampling import SMOTE
# from preprocess import load_and_preprocess_data

# # Load preprocessed data
# X, y = load_and_preprocess_data()

# # Final split for testing
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Apply SMOTE on training set only
# smote = SMOTE(random_state=42, k_neighbors=1)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Define the Random Forest model
# rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# # Hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None]
# }

# # GridSearch with 5-fold CV
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1_macro',
#     n_jobs=-1,
#     verbose=2
# )

# # Fit GridSearch on the resampled training data
# grid_search.fit(X_train_resampled, y_train_resampled)

# # Best model from grid search
# best_model = grid_search.best_estimator_
# print("\nBest Hyperparameters:", grid_search.best_params_)

# # Evaluate on the test set
# y_pred = best_model.predict(X_test)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=['high', 'medium', 'low']))

# # Save the tuned model
# joblib.dump(best_model, 'ml_model/models/rf_tuned.pkl')
# print("\n✅ Tuned model trained and saved successfully.")

