import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv('ml_model/data/kc1.csv')

    # Drop missing values if any
    df.dropna(inplace=True)

    # Define a new multiclass label using a weighted logic (you can adjust this)
    def assign_multiclass_label(row):
        score = row['loc'] * 0.5 + row['v(g)'] * 0.3 + row['ev(g)'] * 0.2
        if score < 30:
            return 0  # High Quality
        elif score < 100:
            return 1  # Medium Quality
        else:
            return 2  # Low Quality

    df['multiclass_label'] = df.apply(assign_multiclass_label, axis=1)

    # Drop the original binary label if it exists
    if 'defects' in df.columns:
        df.drop(columns=['defects'], inplace=True)

    # Split into features and target
    X = df.drop(columns=['multiclass_label'])
    y = df['multiclass_label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    import numpy as np
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))


    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test
