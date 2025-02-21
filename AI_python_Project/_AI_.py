# Stock data sourced from [www.kaggle.com]
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(csv_path):
    # Load CSV data into a DataFrame
    return pd.read_csv(csv_path)

def prepare_dataset(df):
    # Convert 'Date' to datetime and add 'Trend' (0/1) and 'Year' columns
    df["Date"] = pd.to_datetime(df["Date"])
    df["Trend"] = (df["Close"] > df["Open"]).astype(int)
    df["Year"] = df["Date"].dt.year
    return df

def engineer_features(df):
    # Create shifted features from the previous trading day along with today's Open
    epsilon = 1e-6
    df["Open_prev"] = df["Open"].shift(1)
    df["Close_prev"] = df["Close"].shift(1)
    df["Volume_prev"] = df["Volume"].shift(1)
    df["Daily_Change_prev"] = df["Close_prev"] - df["Open_prev"]
    df["Daily_Return_prev"] = (df["Close_prev"] - df["Open_prev"]) / (df["Open_prev"] + epsilon)
    return df

def sanitize_dataset(df):
    # Replace infinite values with NaN, forward-fill, then fill remaining NaNs with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def evaluate_models(train_df, test_df, features):
    # Separate features and labels
    X_train, y_train = train_df[features], train_df["Trend"]
    X_test, y_test = test_df[features], test_df["Trend"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train three models
    nb_model = GaussianNB()
    pc_model = Perceptron(max_iter=1000, eta0=0.001, tol=1e-3)
    lr_model = LogisticRegression(max_iter=1000, penalty='l2', C=0.0001)

    nb_model.fit(X_train_scaled, y_train)
    pc_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)

    # Predictions for each model
    preds_nb = nb_model.predict(X_test_scaled)
    preds_pc = pc_model.predict(X_test_scaled)
    preds_lr = lr_model.predict(X_test_scaled)

    # Accuracy, Precision, Recall, and F1 for Logistic Regression
    print("\n=== Logistic Regression ===")
    acc_lr = accuracy_score(y_test, preds_lr)
    prec_lr = precision_score(y_test, preds_lr)
    rec_lr = recall_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr)
    print(f"Accuracy:  {acc_lr:.4f}")
    print(f"Precision: {prec_lr:.4f}")
    print(f"Recall:    {rec_lr:.4f}")
    print(f"F1-Score:  {f1_lr:.4f}")

    # Accuracy, Precision, Recall, and F1 for Perceptron
    print("\n=== Perceptron ===")
    acc_pc = accuracy_score(y_test, preds_pc)
    prec_pc = precision_score(y_test, preds_pc)
    rec_pc = recall_score(y_test, preds_pc)
    f1_pc = f1_score(y_test, preds_pc)
    print(f"Accuracy:  {acc_pc:.4f}")
    print(f"Precision: {prec_pc:.4f}")
    print(f"Recall:    {rec_pc:.4f}")
    print(f"F1-Score:  {f1_pc:.4f}")

    # Accuracy, Precision, Recall, and F1 for Naive Bayes
    print("\n=== Naive Bayes ===")
    acc_nb = accuracy_score(y_test, preds_nb)
    prec_nb = precision_score(y_test, preds_nb)
    rec_nb = recall_score(y_test, preds_nb)
    f1_nb = f1_score(y_test, preds_nb)
    print(f"Accuracy:  {acc_nb:.4f}")
    print(f"Precision: {prec_nb:.4f}")
    print(f"Recall:    {rec_nb:.4f}")
    print(f"F1-Score:  {f1_nb:.4f}")

def execute_pipeline(file_path, train_years, test_years):
    df = load_dataset(file_path)
    df = prepare_dataset(df)

    # Filter data by specified training and testing years
    train_df = df[df["Year"].isin(train_years)].copy()
    test_df = df[df["Year"].isin(test_years)].copy()
    if train_df.empty or test_df.empty:
        print("No valid data for these years.")
        return

    # Create features and clean the datasets separately
    train_df = engineer_features(train_df)
    train_df = sanitize_dataset(train_df)
    test_df = engineer_features(test_df)
    test_df = sanitize_dataset(test_df)

    features = [
        "Open", "Open_prev", "Close_prev", "Volume_prev",
        "Daily_Change_prev", "Daily_Return_prev"
    ]
    evaluate_models(train_df, test_df, features)

def main():
    path = r"C:\Users\user\PycharmProjects\AI_python_Project\all_stocks_2006-01-01_to_2018-01-01.csv"

    print("Select the years you want to train the model on (from 2006 to 2017, comma-separated):")
    train_input = input("> ").strip()
    train_years = [int(x.strip()) for x in train_input.split(",")]

    print("Select the years you want to test the model on (from 2006 to 2017, comma-separated):")
    test_input = input("> ").strip()
    test_years = [int(x.strip()) for x in test_input.split(",")]

    execute_pipeline(path, train_years, test_years)


main()
