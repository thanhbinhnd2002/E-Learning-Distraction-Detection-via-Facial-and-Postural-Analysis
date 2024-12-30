import cv2
import numpy as np
import pandas as pd
import joblib
import subprocess
import os
import time

# === Đường dẫn ===
openface_path = "/home/binh/Workspace/OpenFace/build/bin/FeatureExtraction"
model_path = "checkpoints/model_RandomForest_2.pkl"
scaler_path = "checkpoints/scaler_2.pkl"

# === Tải mô hình và scaler ===
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Mô hình và Scaler đã được tải thành công.")
except FileNotFoundError as e:
    print(f"Lỗi: {e}")
    exit(1)

# === Hàm trích xuất đặc trưng từ khung hình ===
def extract_features_from_frame(frame, temp_video_path="temp_video.avi", output_csv_path="processed/temp_output.csv"):
    # Ghi frame vào một video tạm thời
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, 30, (width, height))
    out.write(frame)
    out.release()

    # Sử dụng OpenFace để trích xuất đặc trưng
    command = f"{openface_path} -f {temp_video_path} -of {output_csv_path}"
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = pd.read_csv(output_csv_path)
        return data
    except Exception as e:
        print(f"Lỗi khi chạy OpenFace: {e}")
        return None

# === Hàm dự đoán từ đặc trưng ===
def predict_from_features(data, scaler, model):
    try:
        raw_columns_to_keep = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                               'pose_Tx', 'pose_Ty', 'pose_Tz', 'AU06_r', 'AU45_r']
        raw_features = data[raw_columns_to_keep].mean(axis=0).values.reshape(1, -1)
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)
        return prediction[0]
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return None

# === Hàm chính ===
def main():
    cap = cv2.VideoCapture(0)  # Mở webcam
    if not cap.isOpened():
        print("Không thể mở webcam.")
        exit(1)

    print("Đang khởi động realtime...")
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam.")
            break

        # Trích xuất đặc trưng từ khung hình
        features = extract_features_from_frame(frame)

        if features is not None:
            # Dự đoán trạng thái
            prediction = predict_from_features(features, scaler, model)
            if prediction is not None:
                # Hiển thị trạng thái trên khung hình
                label = "Tập Trung" if prediction == 0 else "Mất Tập Trung"
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Hiển thị khung hình
        cv2.imshow("Realtime Distraction Detection", frame)

        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
