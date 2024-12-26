import cv2
import numpy as np
import pandas as pd
import subprocess
import os
import joblib

# === Đường dẫn đến công cụ OpenFace và các mô hình ===
openface_path = "/home/binh/Workspace/OpenFace/build/bin/FeatureExtraction"
output_csv_path = "processed/output_features.csv"
model_path = "checkpoints/model_RandomForest.pkl"
scaler_path = "checkpoints/scaler.pkl"


output_video_path = "./Result/video.mp4"

# === Tải mô hình và scaler ===
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Mô hình đã được tải từ: {model_path}")
    print(f"Scaler đã được tải từ: {scaler_path}")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy mô hình hoặc scaler. Vui lòng chạy mã huấn luyện trước!")
    exit(1)

# === Hàm trích xuất đặc trưng từ video sử dụng OpenFace ===
def extract_features_with_openface(video_path):
    try:
        # Chạy OpenFace để trích xuất đặc trưng
        command = f"{openface_path} -f {video_path} -of {output_csv_path}"
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Đọc file CSV đầu ra từ OpenFace
        data = pd.read_csv(output_csv_path)

        columns_to_keep = ['frame', 'timestamp', 'gaze_angle_x', 'gaze_angle_y',
                   'pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz',
                   'AU06_r', 'AU45_r']

        # Giữ lại các cột cần thiết
        features = data[columns_to_keep]
        features_scaled = scaler.transform(features)


        # Sử dụng cửa sổ thời gian 5 giây (150 frame nếu 30 FPS)
        window_size = 150
        fps = 30  # Số frame mỗi giây
        
        features = data.columns[2:]
        processed_data = []
        for i in range(0, len(features_scaled) - window_size + 1, window_size):
            window = features_scaled[i:i + window_size]
            mean_features = window.mean(axis=0)  # Tính trung bình cho cửa sổ
            max_features = window.max(axis=0)   # Tính giá trị lớn nhất cho cửa sổ
            min_features = window.min(axis=0)   # Tính giá trị nhỏ nhất cho cửa sổ
            std_features = window.std(axis=0)   # Tính độ lệch chuẩn cho cửa sổ

        # Kết hợp tất cả đặc trưng
        combined_features = np.concatenate([mean_features, max_features, min_features, std_features])
        processed_data.append(combined_features)

        columns = (
            [f"mean_{feat}" for feat in features] +
            [f"max_{feat}" for feat in features] +
            [f"min_{feat}" for feat in features] +
            [f"std_{feat}" for feat in features]
    )

        summary_df = pd.DataFrame([processed_data], columns=columns)
        summary_csv_path = "processed/summary_features.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Tính toán đặc trưng đã lưu tại {summary_csv_path}")

        return summary_csv_path
    except subprocess.CalledProcessError as e:
        print(f"Lỗi OpenFace: {e}")
    except Exception as e:
        print(f"Lỗi xử lý đặc trưng: {e}")
    return None

# === Hàm dự đoán từ file CSV đặc trưng ===
def predict_from_features(feature_csv_path):
    try:
        # Đọc file CSV đặc trưng
        features = pd.read_csv(feature_csv_path)

        # # Lọc và sắp xếp các cột theo danh sách đã lưu
        # with open("checkpoints/selected_columns.txt", "r") as f:
        #     selected_columns = f.read().splitlines()
        # features = features[selected_columns]  # Chỉ giữ các cột khớp

        # Chuẩn hóa dữ liệu
        scaled_features = scaler.transform(features)

        # Dự đoán
        predictions = model.predict(scaled_features)
        return predictions[0]  # Vì chỉ có một dòng đặc trưng
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
    return None

# === Xử lý video và tạo đầu ra ===
def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Không thể mở video. Vui lòng kiểm tra đường dẫn!")
        return

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo writer để lưu video đầu ra
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print("Đang xử lý video...")

    # Trích xuất đặc trưng từ OpenFace và tạo file CSV tóm tắt
    summary_csv_path = extract_features_with_openface(input_video_path)
    if summary_csv_path is None:
        print("Không thể trích xuất đặc trưng từ video.")
        return

    # Dự đoán từ file CSV tóm tắt
    prediction = predict_from_features(summary_csv_path)
    if prediction is None:
        print("Không thể dự đoán từ đặc trưng.")
        return

    # Kết quả dự đoán
    result = "Tập Trung" if prediction == 0 else "Mất Tập Trung"
    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hoàn thành xử lý video.")
            break

        # Resize frame về kích thước tiêu chuẩn
        frame = cv2.resize(frame, (640, 480))

        # Hiển thị kết quả trên frame
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Ghi frame vào video đầu ra
        out.write(frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    print(f"Video đầu ra đã được lưu tại {output_video_path}")

if __name__ == "__main__":
    input_video_path = "/home/binh/Workspace/data/data_science/data_raw_1/video_40"  # Thay bằng đường dẫn video của bạn
    process_video(input_video_path)
