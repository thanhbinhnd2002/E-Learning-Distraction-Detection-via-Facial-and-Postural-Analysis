import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import subprocess

def run_openface_on_video(video_path, output_csv):
    """
    Chạy OpenFace trên toàn bộ video và lưu đặc trưng vào file CSV.
    Args:
        video_path (str): Đường dẫn tới file video.
        output_csv (str): Đường dẫn file CSV đầu ra của OpenFace.
    """

    openface_path = "/home/binh/Workspace/projects/OpenFace/build/bin/FeatureExtraction"  # Đường dẫn đến OpenFace FeatureExtraction

    # Chạy OpenFace
    command = f"{openface_path} -f {video_path} -of {output_csv} -2Dfp -pose -aus"
    print(f"Chạy OpenFace trên video: {video_path}")
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"OpenFace đã hoàn thành. Kết quả được lưu tại: {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy OpenFace: {e}")

def process_openface_csv(csv_file, scaler, model, output_csv):
    """
    Xử lý đặc trưng từ file CSV do OpenFace tạo ra và dự đoán trạng thái.
    Args:
        csv_file (str): Đường dẫn tới file CSV của OpenFace.
        scaler: Scaler đã lưu từ huấn luyện.
        model: Mô hình đã huấn luyện.
        output_csv (str): Đường dẫn file kết quả đầu ra (CSV).
    """
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_file)

    # Lọc các đặc trưng đã sử dụng khi huấn luyện
    TRAIN_FEATURES = ["pose_Rx", "pose_Ry", "pose_Rz", "AU06_r", "AU12_r", "AU45_r"]
    #TRAIN_FEATURES = ["face_x", "face_y", "face_z","face_h","face_con","pose_x","pose_y"]
    features = df[TRAIN_FEATURES]

    # Chuẩn hóa dữ liệu
    features_scaled = scaler.transform(features)

    # Cửa sổ thời gian và dự đoán
    window_size = 300  # 10 giây với 30 FPS
    predictions = []
    for i in range(0, len(features_scaled) - window_size + 1, window_size):
        window = features_scaled[i:i + window_size]
        mean_features = window.mean(axis=0)

        # Chuyển mean_features thành DataFrame với đúng tên cột
        mean_features_df = pd.DataFrame([mean_features], columns=TRAIN_FEATURES)

        prediction = model.predict(mean_features_df.values.reshape(1, -1))[0]

        start_time = i / 30  # Tính thời gian bắt đầu (giả sử FPS = 30)
        end_time = (i + window_size) / 30
        predictions.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "prediction": "Tập Trung" if prediction == 0 else "Mất Tập Trung"
        })

    # Lưu kết quả ra file CSV
    result_df = pd.DataFrame(predictions)
    result_df.to_csv(output_csv, index=False)
    print(f"Kết quả đã được lưu tại: {output_csv}")

if __name__ == "__main__":
    # Đường dẫn tới video và file CSV đầu ra
    # video_path = "/home/binh/Workspace/Deeplearning4CV/data/Driver/video10.mp4"
    video_path ="/home/binh/Workspace/projects/OpenFace/build/bin/Input/video3.mp4"
    openface_output_csv = "processed/openface_features.csv"  # File CSV đầu ra của OpenFace
    prediction_output_csv = "Result/predictions_output_0.csv"  # File kết quả dự đoán

    # Chạy OpenFace để trích xuất đặc trưng
    run_openface_on_video(video_path, openface_output_csv)

    # Tải scaler và model
    scaler = joblib.load("checkpoints/scaler.pkl")
    model = joblib.load("checkpoints/model_XGBoost.pkl")

    # Xử lý file CSV từ OpenFace và dự đoán
    process_openface_csv(openface_output_csv, scaler, model, prediction_output_csv)
