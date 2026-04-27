import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Bước 1: Đọc dữ liệu
print("Đang đọc dữ liệu...")
df_list = []
if os.path.exists("data/data_kaggle.csv"):
    df_list.append(pd.read_csv("data/data_kaggle.csv"))
if os.path.exists("data/data_custom.csv"):
    df_list.append(pd.read_csv("data/data_custom.csv"))

if not df_list:
    raise FileNotFoundError("Không tìm thấy file dữ liệu nào! Hãy chạy collect_data.py hoặc extract_from_images.py trước.")

df = pd.concat(df_list, ignore_index=True)

if "label" not in df.columns:
    # Tự thêm header vào
    cols = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    cols.append("label")
    df.columns = cols
    print("Đã tự động thêm header!")

# --- Data Augmentation: Lật ngược trục X để học tay trái ---
print("Đang tạo thêm dữ liệu lật ngược cho tay trái...")
df_left = df.copy()
x_cols = [f"x{i}" for i in range(21)]
for col in x_cols:
    if col in df_left.columns:
        df_left[col] = df_left[col] * -1

df = pd.concat([df, df_left], ignore_index=True)
# -------------------------------------------------------------

print(f"Columns: {df.columns.tolist()[-3:]}")
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
print("\nĐang train model với GridSearchCV (Tìm tham số tốt nhất)...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Train xong! Tham số tốt nhất tìm được: {grid_search.best_params_}")
model = grid_search.best_estimator_

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