import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_tabular import LimeTabularExplainer
import shap

# Function to load dataset and encode the target column
def load_dataset(path=None):
    # Define column names
    col = [
        "duration",
        "total_fiat",
        "total_biat",
        "min_fiat",
        "min_biat",
        "max_fiat",
        "max_biat",
        "mean_fiat",
        "mean_biat",
        "flowPktsPerSecond",
        "flowBytesPerSecond",
        "min_flowiat",
        "max_flowiat",
        "mean_flowiat",
        "std_flowiat",
        "min_active",
        "mean_active",
        "max_active",
        "std_active",
        "min_idle",
        "mean_idle",
        "max_idle",
        "std_idle",
        "class1"
    ]

    # Load dataset with column names
    df = pd.read_csv(r"C:\Users\dhanu\OneDrive\Desktop\jup\ml_lab1\2016_dataset\combine.txt", names=col)
    print("Shape of df before removing NaN values:", df.shape)
    df = df.dropna()
    print("Shape of df after removing NaN values:", df.shape)

    # Encode target column if categorical
    target_col = "class1"
    if df[target_col].dtype == 'object':
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    return df, target_col


# A1: Feature correlation analysis using heatmap
def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Preprocessing features for ML models
def preprocess_features(df, target):
    X = df.drop(columns=[target])
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    return X

# A2 & A3: PCA transformation for dimensionality reduction
def run_pca(X_train, X_test, variance_ratio):
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    pca = PCA(n_components=variance_ratio)
    X_pca_train = pca.fit_transform(X_scaled_train)
    X_pca_test = pca.transform(X_scaled_test)
    return X_pca_train, X_pca_test, pca

# Model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test, label="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{label} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    return acc

# A4: Sequential Feature Selection for feature reduction
def run_fast_sequential_feature_selection(model, X_train, y_train, n_features=10):
    sample_size = min(5000, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction='forward',
        n_jobs=-1
    )
    sfs.fit(X_sample, y_sample)
    selected_idx = sfs.get_support(indices=True)
    return selected_idx, sfs

# A5: Explainability using LIME
def explain_with_lime(model, X_train, X_test, feature_names):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[f"Class {i}" for i in np.unique(model.predict(X_train))],
        mode="classification"
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=5)
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

# A5: Explainability using SHAP
def explain_with_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=True)

def main():
    df, target = load_dataset(r"C:\Users\dhanu\OneDrive\Desktop\jup\ml_lab1\lab3\lab3_vpn_nonvpndataset.txt")

    # A1: Feature correlation heatmap
    plot_correlation_heatmap(df)

    X = preprocess_features(df, target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_model = RandomForestClassifier(random_state=42)

    # A2: PCA retaining 99% variance
    X_pca_train_99, X_pca_test_99, _ = run_pca(X_train, X_test, 0.99)
    acc_99 = evaluate_model(base_model, X_pca_train_99, X_pca_test_99, y_train, y_test, "PCA 99%")

    # A3: PCA retaining 95% variance
    X_pca_train_95, X_pca_test_95, _ = run_pca(X_train, X_test, 0.95)
    acc_95 = evaluate_model(base_model, X_pca_train_95, X_pca_test_95, y_train, y_test, "PCA 95%")

    # A4: Sequential feature selection
    selected_idx, _ = run_fast_sequential_feature_selection(base_model, X_train, y_train, n_features=min(10, X_train.shape[1]))
    X_train_sfs = X_train.iloc[:, selected_idx]
    X_test_sfs = X_test.iloc[:, selected_idx]
    selected_features = X_train_sfs.columns.tolist()
    acc_sfs = evaluate_model(base_model, X_train_sfs, X_test_sfs, y_train, y_test, "Sequential FS (Fast)")

    # A5: Explainability using LIME and SHAP
    explain_with_lime(base_model, X_train_sfs.values, X_test_sfs.values, selected_features)
    explain_with_shap(base_model, X_train_sfs, X_test_sfs)

    # Summary of all experiments
    print(f"PCA 99% Accuracy: {acc_99:.4f}")
    print(f"PCA 95% Accuracy: {acc_95:.4f}")
    print(f"Sequential FS (Fast) Accuracy: {acc_sfs:.4f}")

if __name__ == "__main__":
    main()
