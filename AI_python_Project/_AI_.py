import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 1️⃣ נתיב הקובץ – ישירות, בלי בדיקות מיותרות
file_path = r"C:\Users\user\PycharmProjects\AI_python_Project\all_stocks_2006-01-01_to_2018-01-01.csv"

# 🔹 2️⃣ טעינת הנתונים ישירות
print(f"📂 Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# ✅ 3️⃣ יצירת עמודת `Trend`
df["Trend"] = (df["Close"] > df["Open"]).astype(int)
df["Date"] = pd.to_datetime(df["Date"])

# ✅ 4️⃣ Feature Engineering – יצירת תכונות ללא מידע עתידי
epsilon = 1e-6
df["Daily_Change"] = df["Close"] - df["Open"]
df["Daily_Return"] = (df["Close"] - df["Open"]) / (df["Open"] + epsilon)
df["Volume_Change"] = df["Volume"].pct_change().fillna(0)

# ✅ 5️⃣ ניקוי `inf` ו- `NaN`
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ✅ 6️⃣ חלוקת הדאטה לפי שנים – אימון מבוסס זמן
df["Year"] = df["Date"].dt.year
years = sorted(df["Year"].unique())

# ✅ 7️⃣ בחירת תכונות
features = ["Open", "High", "Low", "Close", "Volume", "Daily_Change", "Daily_Return", "Volume_Change"]

# 🔹 8️⃣ Loop לאימון לפי שנים – אימון על שנה אחת ובדיקה על השנה הבאה
for i in range(len(years) - 1):
    train_year = years[i]
    test_year = years[i + 1]

    print(f"\n🔹 Training on {train_year}, Testing on {test_year}")

    # חלוקת הנתונים לשנה אחת לאימון ושנה אחת לבדיקה
    train_data = df[df["Year"] == train_year]
    test_data = df[df["Year"] == test_year]

    X_train, y_train = train_data[features], train_data["Trend"]
    X_test, y_test = test_data[features], test_data["Trend"]

    # ✅ 9️⃣ בדיקות מקדימות
    print(f"🔎 Checking for inf/NaN in {test_year} data:")
    print("Inf values:", np.isinf(X_test).sum())
    print("NaN values:", X_test.isna().sum())

    # ✅ 🔟 נורמליזציה
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ 1️⃣1️⃣ אימון מודלים
    nb_model = GaussianNB()
    perceptron_model = Perceptron(max_iter=1000, eta0=0.001, tol=1e-3)
    logistic_model = LogisticRegression(max_iter=1000, penalty='l2', C=0.0001)

    nb_model.fit(X_train_scaled, y_train)
    perceptron_model.fit(X_train_scaled, y_train)
    logistic_model.fit(X_train_scaled, y_train)

    # ✅ 1️⃣2️⃣ בדיקת ביצועים
    y_pred_logistic = logistic_model.predict(X_test_scaled)
    y_pred_perceptron = perceptron_model.predict(X_test_scaled)
    y_pred_nb = nb_model.predict(X_test_scaled)

    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logistic):.4f}")
    print(f"Perceptron Accuracy: {accuracy_score(y_test, y_pred_perceptron):.4f}")
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")

    print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_logistic))
    print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_logistic))
