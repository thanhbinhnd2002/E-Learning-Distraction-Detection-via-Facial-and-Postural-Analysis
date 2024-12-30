import subprocess
import pandas as pd
import numpy as np
import joblib
import os

# === Đường dẫn ===
openface_path = "/home/binh/Workspace/OpenFace/build/bin/FeatureExtraction"
output_csv_path = "processed/output_features.csv"
model_path = "checkpoints/model_RandomForest_2.pkl"
scaler_path = "checkpoints/scaler_2.pkl"
result_csv_path = "Result/output_predictions_21_4.csv"

# === Tải mô hình và scaler ===
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Mô hình và Scaler đã được tải thành công.")
except FileNotFoundError as e:
    print(f"Lỗi: {e}")
    exit(1)

# === Hàm trích xuất đặc trưng từ video ===
def extract_features_with_openface(video_path):
    try:
        command = f"{openface_path} -f {video_path} -of {output_csv_path}"
        subprocess.run(command, shell=True, check=True)
        print(f"OpenFace đã hoàn thành. Kết quả lưu tại: {output_csv_path}")
        return output_csv_path
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy OpenFace: {e}")
        return None

# === Hàm xử lý CSV và dự đoán ===
def process_csv_and_predict(input_csv, scaler, model, fps=30):
    try:
        # Đọc file CSV (dữ liệu thô từ OpenFace)
        data = pd.read_csv(input_csv)

        # Chọn các cột đặc trưng thô từ OpenFace
        raw_columns_to_keep = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                               'pose_Tx', 'pose_Ty', 'pose_Tz', 'AU06_r', 'AU45_r']
        raw_features = data[raw_columns_to_keep]

        # Tính toán các đặc trưng trung bình
        window_size = 150  # 5 giây với 30 FPS
        processed_data = []
        timestamps = []

        for i in range(0, len(raw_features) - window_size + 1, window_size):
            window = raw_features.iloc[i:i + window_size]
            mean_features = window.mean(axis=0)

            # Lưu giá trị mean_features
            processed_data.append(mean_features.values)

            # Lưu timestamps
            start_time = i / fps
            end_time = (i + window_size) / fps
            timestamps.append((start_time, end_time))

        # Đặt tên cột theo format mong muốn
        columns = [f"mean_{feat}" for feat in raw_columns_to_keep]

        # Tạo DataFrame từ các đặc trưng đã tính toán
        processed_df = pd.DataFrame(processed_data, columns=columns)
        print(f"Đã xử lý CSV. Kích thước: {processed_df}")
        # Chuẩn hóa dữ liệu
        features_scaled = scaler.transform(processed_df)
        print(f"Đã chuẩn hóa dữ liệu. Kích thước: {features_scaled}")

        # Dự đoán
        predictions = model.predict(features_scaled)
        print(f"Dự đoán: {predictions}")

        # Ghi kết quả vào DataFrame
        result_df = pd.DataFrame({
            "Start_Time": [f"{start:.2f}s" for start, end in timestamps],
            "End_Time": [f"{end:.2f}s" for start, end in timestamps],
            "State": ["Tập Trung" if pred == 0 else "Mất Tập Trung" for pred in predictions]
        })

        # Ghi ra file CSV
        result_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
        print(f"Kết quả đã được ghi vào {result_csv_path}")
        return result_df
    except Exception as e:
        print(f"Lỗi khi xử lý CSV: {e}")
        return None

# === Hàm chính ===
if __name__ == "__main__":
    input_video_path = "/home/binh/Workspace/data/data_science/data_raw_1/video_21/video_21_video_4.mp4"
    fps = 30  # Frame rate của video (FPS)

    # Trích xuất đặc trưng
    csv_path = extract_features_with_openface(input_video_path)
    if csv_path is None:
        print("Lỗi trong quá trình trích xuất đặc trưng.")
        exit(1)

    # Xử lý CSV và dự đoán
    result = process_csv_and_predict(csv_path, scaler, model, fps)
    if result is not None:
        print(result.head())
