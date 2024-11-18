import os
import subprocess

# Đường dẫn đến thư mục chứa video và thư mục lưu kết quả
input_folder = "/home/binh/Workspace/projects/OpenFace/build/bin/Input/"
output_folder = "/home/binh/Workspace/projects/OpenFace/build/bin/output"
openface_path = "/home/binh/Workspace/projects/OpenFace/build/bin/FeatureExtraction"
haar_path = "/home/binh/Workspace/projects/OpenFace/build/bin/model/haarcascade_frontalface_alt.xml"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách tất cả các video trong thư mục input_folder
video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
               if f.lower().endswith(('.mp4', '.avi', '.mkv'))]

print(f"Tìm thấy {len(video_files)} video để xử lý.")

# Kiểm tra nếu không có video nào
if not video_files:
    print("Không có video nào để xử lý.")
    exit()

# Chạy OpenFace để trích xuất đặc trưng cho từng video
for video_path in video_files:
    video_name = os.path.basename(video_path)
    video_folder_name = video_name.replace('.', '_')  # Thay dấu chấm bằng dấu gạch dưới
    print(f"Đang kiểm tra video: {video_name}")

    # Chỉ định đường dẫn đầu ra cho file CSV
    output_dir = os.path.join(output_folder, video_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra nếu file CSV đã tồn tại
    output_csv = os.path.join(output_dir, f"{video_name.split('.')[0]}_of_details.csv")
    if os.path.exists(output_csv):
        print(f"Đã tồn tại file đặc trưng cho video: {video_name}. Bỏ qua video này.")
        continue

    # Lệnh chạy OpenFace
    command = (
        f"{openface_path} "
        f"-f {video_path} "
        f"-out_dir {output_dir} "
        f"-2Dfp -pose -aus "
        f"-fd {haar_path}"
    )
    try:
        # Chạy lệnh OpenFace
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy OpenFace cho video {video_name}: {e}")
        print(e.stderr.decode())
        continue

print("Quá trình trích xuất đặc trưng hoàn tất.")
