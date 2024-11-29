import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# === Đường dẫn đến các mô hình ===
model_path = "checkpoints/model_Random_forest.pkl"
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

# === Bộ nhớ đệm cho cửa sổ thời gian (10 giây, 300 frame) ===
window_size = 300
feature_buffer = deque(maxlen=window_size)

# === Khởi tạo MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === Hàm trích xuất đặc trưng từ MediaPipe ===
def extract_features_with_mediapipe(frame):
    """
    Trích xuất đặc trưng từ khuôn mặt sử dụng MediaPipe Face Mesh.
    Args:
        frame (ndarray): Khung hình từ camera.
    Returns:
        ndarray: Vector đặc trưng hoặc None nếu không phát hiện khuôn mặt.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Chỉ lấy khuôn mặt đầu tiên
        landmarks = results.multi_face_landmarks[0].landmark

        # Trích xuất tọa độ (x, y, z) của các điểm mốc
        features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return features
    return None

# === Hàm tính toán trung bình đặc trưng trong cửa sổ thời gian ===
def calculate_window_mean(buffer):
    """
    Tính giá trị trung bình từ bộ nhớ đệm (cửa sổ frame).
    Args:
        buffer (deque): Bộ nhớ đệm chứa các đặc trưng từ frame.
    Returns:
        ndarray: Vector trung bình của cửa sổ.
    """
    buffer_array = np.array(buffer)
    return np.mean(buffer_array, axis=0)

# === Vòng lặp xử lý video ===
def process_video():
    """
    Xử lý video từ webcam và dự đoán trạng thái tập trung theo thời gian thực.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra kết nối!")
        return

    print("Camera đã bật. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break

        # Trích xuất đặc trưng từ frame
        face_features = extract_features_with_mediapipe(frame)

        if face_features is not None:
            # Thêm đặc trưng vào bộ nhớ đệm
            feature_buffer.append(face_features)

            # Khi đủ 10 giây (300 frame), tính toán trung bình và dự đoán
            if len(feature_buffer) == window_size:
                mean_features = calculate_window_mean(feature_buffer)

                # Chuẩn hóa đặc trưng
                mean_features_scaled = scaler.transform([mean_features])

                # Dự đoán với mô hình
                prediction = model.predict(mean_features_scaled)[0]

                # Hiển thị kết quả trên video
                if prediction == 1:
                    cv2.putText(frame, "Mất Tập Trung", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Tập Trung", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị video
        cv2.imshow("Camera", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đang thoát chương trình...")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# === Chạy chương trình ===
if __name__ == "__main__":
    process_video()
