import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np

# Hàm trích xuất đặc trưng từ khung hình
def extract_features_from_frame(frame):
    """
    Trích xuất đặc trưng từ một khung hình bằng MediaPipe.
    """
    # Khởi tạo Mediapipe
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    pose = mp_pose.Pose()

    # Chuyển khung hình sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Đặc trưng khuôn mặt
    face_x, face_y, face_w, face_h, face_con = 0, 0, 0, 0, 0
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_x = int(bbox.xmin * frame_width)
            face_y = int(bbox.ymin * frame_height)
            face_w = int(bbox.width * frame_width)
            face_h = int(bbox.height * frame_height)
            face_con = detection.score[0]

    # Đặc trưng tư thế (pose)
    pose_x, pose_y, pose_label = 0, 0, "None"
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        pose_x = nose.x * frame_width
        pose_y = nose.y * frame_height

        # Tính toán pose label (Forward, Left, Right, Down)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if abs(left_shoulder.y - right_shoulder.y) < 0.05:  # Đầu ngang
            pose_label = "forward"
        elif left_shoulder.y > right_shoulder.y:  # Quay trái
            pose_label = "left"
        elif left_shoulder.y < right_shoulder.y:  # Quay phải
            pose_label = "right"
        else:
            pose_label = "down"

    # Trả về đặc trưng
    return {
        "face_x": face_x,
        "face_y": face_y,
        "face_w": face_w,
        "face_h": face_h,
        "face_con": face_con,
        "pose": pose_label,
        # "pose_x": pose_x,
        # "pose_y": pose_y,
    }

# Hàm triển khai thời gian thực
def realtime_attention_detection(model_path, scaler_path, encoder_path):
    """
    Triển khai nhận diện tập trung theo thời gian thực bằng camera laptop.
    """
    # Tải mô hình
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    # Mở camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    # Đặt độ phân giải cho camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Resize frame về kích thước 640x480
        frame = cv2.resize(frame, (640, 480))

        # Trích xuất đặc trưng
        feature = extract_features_from_frame(frame)

        # Tạo DataFrame từ đặc trưng
        df = pd.DataFrame([feature])

        # Xử lý dữ liệu: Chuẩn hóa numeric + One-Hot Encoding
        numeric_columns = ['face_x', 'face_y', 'face_w', 'face_h', 'face_con']
        categorical_columns = ['pose']

        # Chuẩn hóa các cột numeric
        df[numeric_columns] = scaler.transform(df[numeric_columns])

        # One-hot encode cột "pose"
        pose_encoded = encoder.transform(df[categorical_columns]).toarray()
        pose_encoded_columns = encoder.get_feature_names_out(categorical_columns)

        # Ghép lại DataFrame với các cột đã xử lý
        df_encoded = pd.concat([df[numeric_columns], pd.DataFrame(pose_encoded, columns=pose_encoded_columns)], axis=1)

        # Đảm bảo đồng bộ cột với mô hình
        expected_columns = model.named_steps['preprocessor'].get_feature_names_out()
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Giá trị mặc định là 0

        df_encoded = df_encoded[expected_columns]

        # Dự đoán trạng thái
        prediction = model.predict(df_encoded)[0]

        # Hiển thị trạng thái
        status = "Tập trung" if prediction == 0 else "Mất tập trung"
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)

        # Hiển thị khung hình và trạng thái
        cv2.putText(frame, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow("Real-Time Attention Detection", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đường dẫn mô hình và các file scaler/encoder
    model_path = "Model/trained_model_SVM.pkl"
    scaler_path = "Model/scaler.pkl"
    encoder_path = "Model/encoder.pkl"

    # Triển khai nhận diện thời gian thực
    realtime_attention_detection(model_path, scaler_path, encoder_path)
