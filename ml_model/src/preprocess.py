import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

def load_and_preprocess_data(show_table=False, k_features=10):
    # Load dataset
    df = pd.read_csv('ml_model/data/kc1.csv')
    df.dropna(inplace=True)

    # Assign new labels based on weighted metrics
    def assign_multiclass_label(row):
        score = row['loc'] * 0.5 + row['v(g)'] * 0.3 + row['ev(g)'] * 0.2
        if score < 5:
            return 0  # High Quality
        elif score < 15:
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

    # === Feature Selection (SelectKBest with chi2) ===
    selector = SelectKBest(score_func=chi2, k=k_features)
    X_selected = selector.fit_transform(X_scaled, y)

    # Get selected feature names + their scores
    feature_scores = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)

    print("\nTop Features based on chi2 test:")
    print(feature_scores.head(k_features))

    # Convert scaled selected data back to DataFrame
    selected_features = X.columns[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    X_selected_df['multiclass_label'] = y.values  # Add label back for display


    if show_table:
        print("\nPreview of scaled + selected features:")
        print(X_selected_df.head())

    return X_selected, y, selected_features

if __name__ == "__main__":
    # Example usage
    X, y, selected_features = load_and_preprocess_data(show_table=True, k_features=10)
    print("\nSelected Features for training:")
    print(selected_features)