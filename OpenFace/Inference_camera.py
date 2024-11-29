import cv2
import numpy as np
import joblib
from collections import deque
import subprocess
from threading import Thread
import queue

# === Đường dẫn đến công cụ OpenFace và các mô hình ===
openface_path = "/home/binh/Workspace/projects/OpenFace/build/bin/FeatureExtraction"
temp_frame_path = "temp_frame.jpg"
temp_output_path = "processed/temp_output.csv"
model_path = "checkpoints/model_XGBoost.pkl"
scaler_path = "checkpoints/scaler.pkl"

# === Tải mô hình và scaler ===
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Mô hình đã được tải từ: {model_path}")
    print(f"Scaler đã được tải từ: {scaler_path}")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy mô hình hoặc scaler. Vui lòng chạy mã huấn luyện trước!")
    exit(1)

# === Bộ nhớ đệm và hàng đợi ===
feature_buffer = deque(maxlen=300)  # Lưu trữ đặc trưng trong cửa sổ thời gian (10 giây, 300 frame)
frame_queue = queue.Queue()  # Hàng đợi lưu frame từ camera
feature_queue = queue.Queue()  # Hàng đợi lưu đặc trưng đã xử lý từ OpenFace

# === Hàm trích xuất đặc trưng từ frame sử dụng OpenFace ===
def extract_features_with_openface(frame):
    try:
        # Lưu frame tạm thời
        cv2.imwrite(temp_frame_path, frame)

        # Chạy OpenFace để trích xuất đặc trưng
        command = f"{openface_path} -f {temp_frame_path} -of {temp_output_path} -2Dfp -pose -aus"
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Đọc đặc trưng từ file CSV
        with open(temp_output_path, "r") as file:
            lines = file.readlines()
            if len(lines) > 1:  # Dòng đầu là tiêu đề, cần ít nhất một dòng dữ liệu
                features = np.array(lines[1].strip().split(",")[2:], dtype=np.float32)  # Bỏ frame và timestamp
                return features
    except subprocess.CalledProcessError as e:
        print(f"Lỗi OpenFace: {e}")
    except Exception as e:
        print(f"Lỗi xử lý OpenFace output: {e}")
    return None

# === Luồng xử lý frame (trích xuất đặc trưng với OpenFace) ===
def process_frame_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:  # Tín hiệu thoát
            break

        # Trích xuất đặc trưng từ frame
        face_features = extract_features_with_openface(frame)
        if face_features is not None:
            feature_queue.put(face_features)

# === Luồng dự đoán (dùng model để đưa ra kết quả) ===
def prediction_thread():
    while True:
        features = feature_queue.get()
        if features is None:  # Tín hiệu thoát
            break

        # Lưu đặc trưng vào bộ nhớ đệm
        feature_buffer.append(features)

        # Khi đủ 300 frame (10 giây), tính toán trung bình và dự đoán
        if len(feature_buffer) == feature_buffer.maxlen:
            mean_features = np.mean(feature_buffer, axis=0)
            mean_features_scaled = scaler.transform([mean_features])
            prediction = model.predict(mean_features_scaled)[0]

            # Hiển thị kết quả
            result = "Tập Trung" if prediction == 0 else "Mất Tập Trung"
            print(f"Kết quả dự đoán: {result}")

# === Vòng lặp xử lý video ===
def process_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra kết nối!")
        return

    print("Camera đã bật. Nhấn 'q' để thoát.")

    # Tạo các luồng xử lý
    thread_frame = Thread(target=process_frame_thread, daemon=True)
    thread_predict = Thread(target=prediction_thread, daemon=True)
    thread_frame.start()
    thread_predict.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break

        # Resize frame về kích thước tiêu chuẩn
        frame = cv2.resize(frame, (640, 480))

        # Gửi frame vào hàng đợi xử lý
        frame_queue.put(frame)

        # Hiển thị frame trên màn hình
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đang thoát chương trình...")
            break

    # Kết thúc chương trình
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)  # Tín hiệu thoát cho luồng xử lý frame
    feature_queue.put(None)  # Tín hiệu thoát cho luồng dự đoán
    thread_frame.join()
    thread_predict.join()

if __name__ == "__main__":
    process_video()
