import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Bước 1: Đọc dữ liệu
print("Đang đọc dữ liệu...")
df = pd.read_csv("data/data.csv")

# Bước 2: Tách X và y
X = df.drop("label", axis=1).values
y = df["label"].values

print(f"Tổng mẫu: {len(X)}")
print(f"Số chữ cái: {len(set(y))}")

# Bước 3: Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)} mẫu")
print(f"Test:  {len(X_test)} mẫu")

# Bước 4: Train Random Forest
print("\nĐang train model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Train xong!")

# Bước 5: Đánh giá
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác: {accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))

# Bước 6: Lưu model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Đã lưu model vào model/model.pkl")