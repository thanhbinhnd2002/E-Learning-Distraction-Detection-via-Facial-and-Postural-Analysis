import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến file CSV đã được trích xuất đặc trưng
input_csv = "/home/binh/Workspace/projects/OpenFace/build/bin/processed_features.csv"

# Đọc file CSV
print("Đang đọc file CSV...")
data = pd.read_csv(input_csv)
print(f"Đã đọc {len(data)} dòng dữ liệu.")

# Lọc các cột cần thiết
columns_to_keep = ['mean_pose_Rx', 'mean_pose_Ry', 'mean_pose_Rz', 'mean_AU06_r', 'mean_AU12_r', 'mean_AU45_r',
                   'std_pose_Rx', 'std_pose_Ry', 'std_pose_Rz', 'std_AU06_r', 'std_AU12_r', 'std_AU45_r',
                   'max_pose_Rx', 'max_pose_Ry', 'max_pose_Rz', 'max_AU06_r', 'max_AU12_r', 'max_AU45_r',
                   'min_pose_Rx', 'min_pose_Ry', 'min_pose_Rz', 'min_AU06_r', 'min_AU12_r', 'min_AU45_r', 'label']

# Kiểm tra nếu các cột cần thiết có trong dữ liệu
if not all(col in data.columns for col in columns_to_keep):
    print("Thiếu các cột cần thiết trong dữ liệu. Hãy kiểm tra lại file CSV.")
    exit()

# Lọc các cột và nội suy giá trị thiếu
print("Tiền xử lý dữ liệu...")
data = data[columns_to_keep]
data.interpolate(method='linear', inplace=True)

# Loại bỏ ngoại lệ bằng Isolation Forest
features = columns_to_keep[:-1]  # Không bao gồm cột 'label'
clf = IsolationForest(contamination=0.01, random_state=42)
data['is_outlier'] = clf.fit_predict(data[features])
data = data[data['is_outlier'] == 1].drop(columns=['is_outlier'])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
y = data['label']

# Chia tập huấn luyện và tập kiểm tra
print("Chia dữ liệu thành tập huấn luyện và kiểm tra...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Số lượng mẫu trong tập huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu trong tập kiểm tra: {len(X_test)}")

# Huấn luyện mô hình Random Forest
print("Huấn luyện mô hình Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
print("Dự đoán trên tập kiểm tra...")
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Đánh giá mô hình:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Hiển thị ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Dự đoán trên một mẫu mới (ví dụ)
print("Dự đoán trên một mẫu mới:")
new_sample = X_test[0].reshape(1, -1)
prediction = model.predict(new_sample)
print("Dự đoán:", "Mất tập trung" if prediction[0] == 1 else "Tập trung")
