import cv2
import mediapipe as mp
import pandas as pd

def extract_features_from_frame(frame):
    """
    Trích xuất đặc trưng từ một khung hình bằng MediaPipe và chuyển sang tọa độ pixel.
    """
    # Khởi tạo MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Chuyển khung hình sang RGB (MediaPipe yêu cầu định dạng RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape  # Lấy kích thước khung hình

    # Phát hiện khuôn mặt
    face_x, face_y, face_w, face_h, face_con = 0, 0, 0, 0, 0
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Chuyển tỷ lệ tương đối thành tọa độ pixel
            face_x = int(bbox.xmin * frame_width)
            face_y = int(bbox.ymin * frame_height)
            face_w = int(bbox.width * frame_width)
            face_h = int(bbox.height * frame_height)
            face_con = detection.score[0]

    # Trả về đặc trưng
    return {
        "face_x": face_x,
        "face_y": face_y,
        "face_w": face_w,
        "face_h": face_h,
        "face_con": face_con,
    }

def process_video_and_save_to_csv(video_path, output_csv):
    """
    Đọc video, trích xuất đặc trưng từ từng khung hình và lưu vào file CSV.
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy số khung hình trên giây
    features = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý từng khung hình
        if frame_count % frame_rate == 0:  # Lấy mỗi 1 khung hình mỗi giây
            feature = extract_features_from_frame(frame)
            feature["time"] = frame_count // frame_rate  # Thêm thông tin thời gian (giây)
            features.append(feature)

        frame_count += 1

    cap.release()

    # Lưu đặc trưng vào file CSV
    df = pd.DataFrame(features)
    df.to_csv(output_csv, index=False)
    print(f"Đã lưu đặc trưng vào file: {output_csv}")

if __name__ == "__main__":
    # Đường dẫn video và file CSV đầu ra
    video_path = "/home/binh/Workspace/Deeplearning4CV/data/Driver/video2.mp4"  # Thay bằng đường dẫn tới video của bạn
    output_csv = "test/output_features.csv"  # Tên file CSV đầu ra

    # Xử lý video và lưu kết quả
    process_video_and_save_to_csv(video_path, output_csv)
