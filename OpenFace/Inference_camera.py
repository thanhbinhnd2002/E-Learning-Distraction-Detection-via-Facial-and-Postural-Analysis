import cv2
import numpy as np
import pandas as pd
import subprocess
import os
import joblib

# === Đường dẫn ===
openface_path = "/home/binh/Workspace/OpenFace/build/bin/FeatureExtraction"
output_csv_path = "processed/output_features.csv"
model_path = "checkpoints/model_RandomForest_2.pkl"
scaler_path = "checkpoints/scaler_2.pkl"
output_video_path = "./Result/video_annotated.mp4"

# === Tải mô hình và scaler ===
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Mô hình và Scaler đã được tải thành công.")
except FileNotFoundError as e:
    print(f"Lỗi: {e}")
    exit(1)

# === Hàm trích xuất đặc trưng từ video ===
def extract_features_with_openface(video_path):
    try:
        command = f"{openface_path} -f {video_path} -of {output_csv_path}"
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"OpenFace đã hoàn thành. Kết quả lưu tại: {output_csv_path}")
        return output_csv_path
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy OpenFace: {e}")
        return None

# === Hàm xử lý CSV và dự đoán ===
def process_csv_and_predict(input_csv, scaler, model):
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

        for i in range(0, len(raw_features) - window_size + 1, window_size):
            window = raw_features.iloc[i:i + window_size]
            mean_features = window.mean(axis=0)

            # Chỉ sử dụng mean_features
            processed_data.append(mean_features)

        # Tạo DataFrame từ các đặc trưng đã tính toán
        processed_df = pd.DataFrame(processed_data, columns=mean_features.index)

        # Chuẩn hóa dữ liệu
        features_scaled = scaler.transform(processed_df)

        # Dự đoán
        predictions = model.predict(features_scaled)
        return predictions
    except Exception as e:
        print(f"Lỗi khi xử lý CSV: {e}")
        return None

# === Hàm tạo video với nhãn ===
def create_annotated_video(input_video, output_video, predictions, fps):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    prediction_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hiển thị nhãn trên mỗi frame
        if prediction_index < len(predictions):
            label = "Tập Trung" if predictions[prediction_index] == 0 else "Mất Tập Trung"
            color = (0, 255, 0) if predictions[prediction_index] == 0 else (0, 0, 255)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Ghi frame vào video
        out.write(frame)

        frame_count += 1
        if frame_count % 150 == 0:  # Cập nhật nhãn mỗi 150 frame (5 giây)
            prediction_index += 1

    cap.release()
    out.release()
    print(f"Video đã được lưu tại: {output_video}")

# === Hàm chính ===
if __name__ == "__main__":
    input_video_path = "/home/binh/Workspace/data/data_science/data_raw_1/video_40/video_40_video_4.mp4" 
    fps = 30

    # Trích xuất đặc trưng
    csv_path = extract_features_with_openface(input_video_path)
    if csv_path is None:
        print("Lỗi trong quá trình trích xuất đặc trưng.")
        exit(1)

    # Xử lý CSV và dự đoán
    predictions = process_csv_and_predict(csv_path, scaler, model)
    if predictions is None:
        print("Lỗi trong quá trình dự đoán.")
        exit(1)

    # Tạo video với nhãn
    create_annotated_video(input_video_path, output_video_path, predictions, fps)
