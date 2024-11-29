import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

def train_model(input_csv, model_output_path, scaler_output_path):
    # Đọc file CSV
    data = pd.read_csv(input_csv)

    # **Chuyển đổi giá trị của cột 'pose'**
    pose_mapping = {'forward': 1, 'left': 2, 'right': 3, 'down': 4}
    data['pose'] = data['pose'].map(pose_mapping)

    # Chỉ giữ lại các thuộc tính cần thiết
    features = ['face_x', 'face_y', 'face_w', 'face_h', 'face_con', 'pose']
    target = 'label'  # Cột nhãn mục tiêu (0: Tập trung, 1: Mất tập trung)
    data = data[features + [target]]

    # Xử lý dữ liệu
    X = data[features]
    y = data[target]

    # **Tách các cột numeric và categorical**
    numeric_columns = ['face_x', 'face_y', 'face_w', 'face_h', 'face_con', 'pose']

    # Khởi tạo ColumnTransformer cho dữ liệu numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns)  # Chuẩn hóa tất cả các cột số
        ]
    )

    # Khởi tạo mô hình SVM
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True, random_state=42))
    ])

    # Chia dữ liệu thành tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)

    # **Hiển thị kết quả đánh giá**
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # **Hiển thị ma trận nhầm lẫn**
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.named_steps['classifier'].classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix SVM")
    plt.show()

    # **Đánh giá độ chính xác**
    accuracy = model.score(X_test, y_test)
    print(f"Độ chính xác trên tập test: {accuracy:.2f}")

    # **Lưu mô hình**
    joblib.dump(model, model_output_path)
    print(f"Đã lưu mô hình tại: {model_output_path}")

    # **Lưu scaler riêng biệt**
    # Tách scaler từ preprocessor
    scaler = preprocessor.named_transformers_['num']
    joblib.dump(scaler, scaler_output_path + "/scaler.pkl")
    print(f"Đã lưu scaler tại: {scaler_output_path}/scaler.pkl")

if __name__ == "__main__":
    # Đường dẫn file CSV đầu vào và nơi lưu mô hình
    input_csv = "/home/binh/Downloads/attention_detection_dataset_v1.csv"  # Thay bằng đường dẫn tới file CSV của bạn
    model_output_path = "Model/trained_model_SVM.pkl"
    scaler_output_path = "Model"  # Nơi lưu scaler và encoder

    # Huấn luyện mô hình
    train_model(input_csv, model_output_path, scaler_output_path)
