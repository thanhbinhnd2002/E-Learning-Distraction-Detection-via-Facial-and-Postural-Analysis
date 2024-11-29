import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# === Hàm hỗ trợ ===

def load_data(csv_file):
    """
    Đọc dữ liệu từ file CSV và tách đặc trưng và nhãn.
    Args:
        csv_file (str): Đường dẫn đến file CSV.
    Returns:
        pd.DataFrame, pd.Series: Đặc trưng (X) và nhãn (y).
    """


    data = pd.read_csv(csv_file)

    X = data.drop(columns=["label","no_of_hand","phone","phone_x","phone_y","phone_w","phone_h","phone_con","no_of_face","pose"],)
    y = data["label"]
    return X, y

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Vẽ confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# === Huấn luyện mô hình ===

def train_model(X, y, model_type="RandomForest", test_size=0.2, random_state=42, checkpoint_dir="checkpoints"):
    """
    Huấn luyện mô hình từ dữ liệu đã chuẩn bị.
    Args:
        X (pd.DataFrame): Đặc trưng.
        y (pd.Series): Nhãn.
        model_type (str): Loại mô hình ("RandomForest" hoặc "XGBoost").
        test_size (float): Tỷ lệ dữ liệu kiểm tra.
        random_state (int): Giá trị random seed.
        checkpoint_dir (str): Thư mục lưu mô hình.
    """
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Chọn mô hình
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "XGBoost":
        import xgboost as xgb
        model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=random_state)
    else:
        raise ValueError("Mô hình không hợp lệ. Chọn 'RandomForest' hoặc 'XGBoost'.")

    #chuẩn hóa dữ liệu
    scale  = StandardScaler()
    scaler = scale.fit(X_train)

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy trên tập kiểm tra: {accuracy:.4f}")

    # Vẽ confusion matrix
    class_names = y.unique()
    plot_confusion_matrix(y_test, y_pred, class_names)

    # Lưu mô hình
    model_filename = f"{checkpoint_dir}/model2.pkl"
    scaler_filename = f"{checkpoint_dir}/scaler2.pkl"

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Mô hình đã được lưu tại {model_filename}")


    return model

# === Hàm chính ===

if __name__ == "__main__":
    # Đường dẫn đến file CSV chứa dữ liệu huấn luyện
    #csv_file = "/home/binh/Output/Final_Data/processed_features.csv"  # Thay bằng đường dẫn tới file CSV của bạn
    csv_file = "/home/binh/Downloads/attention_detection_dataset_v1.csv"

    # Đọc dữ liệu
    X, y = load_data(csv_file)

    # Huấn luyện mô hình
    model = train_model(X, y, model_type="RandomForest", checkpoint_dir="checkpoints")
