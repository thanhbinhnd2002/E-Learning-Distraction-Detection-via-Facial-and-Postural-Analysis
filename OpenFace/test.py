import subprocess
import os

# === Đường dẫn đến OpenFace và file CSV đầu ra ===
openface_path = "/home/binh/Workspace/OpenFace/build/bin/FeatureExtraction"  # Đường dẫn đến OpenFace FeatureExtraction
output_csv_path = "processed/output_features.csv"  # File CSV đầu ra

def run_openface_on_video(video_path, output_csv):
    """
    Chạy OpenFace để phân tích video và lưu kết quả vào file CSV.
    Args:
        video_path (str): Đường dẫn tới video đầu vào.
        output_csv (str): Đường dẫn tới file CSV đầu ra.
    """
    try:
        # Kiểm tra xem file video có tồn tại không
        if not os.path.exists(video_path):
            print(f"Lỗi: Không tìm thấy video tại {video_path}")
            return

        # Tạo thư mục chứa file CSV nếu chưa tồn tại
        output_dir = os.path.dirname(output_csv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Chạy OpenFace
        command = f"{openface_path} -f {video_path} -of {output_csv} -2Dfp -pose -aus"
        print(f"Đang chạy OpenFace trên video: {video_path}")
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Phân tích hoàn thành. Kết quả được lưu tại: {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy OpenFace: {e}")
    except Exception as e:
        print(f"Lỗi xử lý: {e}")

if __name__ == "__main__":
    # Đường dẫn video đầu vào
    input_video_path = "/home/binh/Workspace/data/data_science/data_raw_1/video_29/video_29_video_3.mp4"  # Thay đường dẫn bằng video của bạn
    run_openface_on_video(input_video_path, output_csv_path)
