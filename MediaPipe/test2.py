import mediapipe as mp
import cv2
# Khởi tạo các mô-đun MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


# Xử lý từng khung hình
def process_frame(frame):
    # Chuyển sang RGB (MediaPipe yêu cầu)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            face_x = bbox.xmin
            face_y = bbox.ymin
            face_w = bbox.width
            face_h = bbox.height
            face_conf = detection.score[0]

    # Phát hiện bàn tay
    hand_results = hands.process(rgb_frame)
    no_of_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

    # Trả về các đặc trưng
    return {
        "face_x": face_x,
        "face_y": face_y,
        "face_w": face_w,
        "face_h": face_h,
        "face_conf": face_conf,
        "no_of_hands": no_of_hands,
    }

