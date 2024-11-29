import cv2
import mediapipe as mp
import pandas as pd
import joblib
import numpy as np

def extract_features_from_frame(frame):
    """
    Trích xuất đặc trưng từ một khung hình bằng MediaPipe.
    """
    # Khởi tạo MediaPipe Face Detection và Pose
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    pose = mp_pose.Pose()

    # Chuyển đổi khung hình sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Đặc trưng khuôn mặt
    face_x, face_y, face_w, face_h, face_con = 0, 0, 0, 0, 0
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_boxgit
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
    }

def process_video_and_predict(video_path, model_path, scaler_path, output_csv):
    """
    Đọc video, chuẩn hóa kích thước, trích xuất đặc trưng và dự đoán nhãn mất tập trung.
    """
    # Tải mô hình và scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Mapping cột pose từ label sang số (giống code train)
    pose_mapping = {'forward': 1, 'left': 2, 'right': 3, 'down': 4}

    # Đọc video
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy số khung hình mỗi giây
    features = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame về kích thước 640x480
        frame = cv2.resize(frame, (640, 480))

        # Chỉ xử lý mỗi 5 giây
        if frame_count % (5 * frame_rate) == 0:
            feature = extract_features_from_frame(frame)

            # Chuyển đổi cột pose thành giá trị số
            feature['pose'] = pose_mapping.get(feature['pose'], 0)  # Nếu không khớp, gán mặc định là 0

            features.append(feature)

        frame_count += 1

    cap.release()

    # Tạo DataFrame từ các đặc trưng
    df = pd.DataFrame(features)

    # Kiểm tra nếu DataFrame trống
    if df.empty:
        print("Không có dữ liệu đầu vào từ video.")
        return

    # Tách các cột numeric
    numeric_columns = ['face_x', 'face_y', 'face_w', 'face_h', 'face_con', 'pose']

    # Chuẩn hóa dữ liệu numeric
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    # Dự đoán nhãn
    predictions = model.predict(df[numeric_columns])
    df['prediction'] = predictions

    # Lưu kết quả vào file CSV
    df.to_csv(output_csv, index=False)
    print(f"Kết quả dự đoán đã lưu tại: {output_csv}")

if __name__ == "__main__":
    # Đường dẫn video và file CSV đầu ra
    video_path = "/home/binh/Workspace/projects/OpenFace/build/bin/Input/video3.mp4"  # Thay bằng đường dẫn tới video của bạn
    model_path = "Model/trained_model_SVM.pkl"
    scaler_path = "Model/scaler.pkl"
    output_csv = "Result/predicted_results_2.csv"

    # Triển khai dự đoán
    process_video_and_predict(video_path, model_path, scaler_path, output_csv)
