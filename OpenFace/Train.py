import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Bỏ cột 'video_name' nếu tồn tại
    if "video_name" in data.columns:
        data = data.drop(columns=["video_name"])
        print("Đã loại bỏ cột 'video_name'.")
    X = data.drop(columns=["Label_1"])
    y = data["Label_1"]
    # Lưu danh sách các cột


    return X, y

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Vẽ confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# === Huấn luyện mô hình ===
def train_model(X, y, model_type="RandomForest", test_size=0.2, random_state=43, checkpoint_dir="checkpoints"):
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

    # selected_columns = list(X_train.columns)
    # with open("checkpoints/selected_columns.txt", "w") as f:
    #     f.write("\n".join(selected_columns))
    #     print("Danh sách cột đã được lưu.")

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Lưu scaler
    scaler_filename = f"{checkpoint_dir}/scaler_2.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler đã được lưu tại {scaler_filename}")

    # Chọn mô hình
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "XGBoost":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    elif model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel="linear", random_state=random_state)
    elif model_type == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_type == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError("Mô hình không hợp lệ. Chọn 'RandomForest' hoặc 'XGBoost'.")

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # === Dự đoán và đánh giá trên tập huấn luyện ===
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average="macro")
    train_recall = recall_score(y_train, y_train_pred, average="macro")
    print(f"\n=== Kết quả trên tập huấn luyện ===")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    plot_confusion_matrix(y_train, y_train_pred, y.unique(), title="Confusion Matrix (Train)")

    # === Dự đoán và đánh giá trên tập kiểm tra ===
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="macro")
    test_recall = recall_score(y_test, y_test_pred, average="macro")
    print(f"\n=== Kết quả trên tập kiểm tra ===")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    plot_confusion_matrix(y_test, y_test_pred, y.unique(), title="Confusion Matrix (Test)")

    # Lưu mô hình
    model_filename = f"{checkpoint_dir}/model_{model_type}_2.pkl"
    joblib.dump(model, model_filename)
    print(f"Mô hình đã được lưu tại {model_filename}")

    return model

# === Hàm arg ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model for classification")
    parser.add_argument("--csv_file", type=str, default="/home/binh/Workspace/data/data_science/test/test_3.csv", help="Path to the input CSV file")
    parser.add_argument("--model_type", type=str, default="SVM",
                        choices=["RandomForest", "XGBoost", "SVM", "Decision Tree", "KNN"],
                        help="Type of model to train")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test data")
    parser.add_argument("--random_state", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the model")
    return parser.parse_args()

# === Hàm chính ===
if __name__ == "__main__":
    args = parse_args()

    # Đọc dữ liệu
    X, y = load_data(args.csv_file)
    print(X)
    print(y)

    # Huấn luyện mô hình
    model = train_model(X, y, model_type=args.model_type, test_size=args.test_size,
                        random_state=args.random_state, checkpoint_dir=args.checkpoint_dir)
