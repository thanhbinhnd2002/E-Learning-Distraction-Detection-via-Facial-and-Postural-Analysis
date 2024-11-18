import os
import shutil

# Đường dẫn đến thư mục Input
input_folder = "/home/binh/Workspace/projects/OpenFace/build/bin/Input/"

# Tạo danh sách chứa đường dẫn đến tất cả các video
video_files = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mkv')):
            video_files.append(os.path.join(root, file))

# Đường dẫn mới cho tất cả các video
output_folder = input_folder  # Đặt tất cả video vào chung thư mục Input
os.makedirs(output_folder, exist_ok=True)

# Đổi tên video
for i, video_path in enumerate(video_files):
    # Lấy phần mở rộng của file (ví dụ: .mp4, .mkv)
    file_extension = os.path.splitext(video_path)[1].lower()
    # Đặt tên mới cho video
    new_name = f"video{i + 1}{file_extension}"
    new_path = os.path.join(output_folder, new_name)

    # Di chuyển và đổi tên video
    try:
        shutil.move(video_path, new_path)
        print(f"Đã đổi tên: {video_path} -> {new_path}")
    except Exception as e:
        print(f"Lỗi khi đổi tên video {video_path}: {e}")

print("Quá trình đổi tên hoàn tất.")
