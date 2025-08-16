import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(show_table=False):
    # Load dataset
    df = pd.read_csv('ml_model/data/kc1.csv')
    df.dropna(inplace=True)

    # Assign new labels based on weighted metrics
    def assign_multiclass_label(row):
        score = row['loc'] * 0.5 + row['v(g)'] * 0.3 + row['ev(g)'] * 0.2
        if score < 30:
            return 0  # High Quality
        elif score < 100:
            return 1  # Medium Quality
        else:
            return 2  # Low Quality

    df['multiclass_label'] = df.apply(assign_multiclass_label, axis=1)

    # Drop original 'defects' label if present
    if 'defects' in df.columns:
        df.drop(columns=['defects'], inplace=True)

    # Separate features and target
    X = df.drop(columns=['multiclass_label'])
    y = df['multiclass_label']

# Apply MinMax scaling (0 to 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled data back to DataFrame with original column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df['multiclass_label'] = y.values  # Add label back for display

    # Optionally display first 5 rows
    if show_table:
        print(X_scaled_df.head())

    return X_scaled, y
