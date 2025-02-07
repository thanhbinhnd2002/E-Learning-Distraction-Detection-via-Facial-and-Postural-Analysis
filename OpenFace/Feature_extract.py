import os
import subprocess


def find_video_files(input_folder):
    """
    Lấy tất cả các file video trong thư mục đầu vào.
    Args:
    - input_folder (str): Đường dẫn đến thư mục chứa video.

    Returns:
    - list: Danh sách các file video tìm được.
    """
    video_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Có thể thêm các định dạng khác
                video_files.append(os.path.join(root, file))
    return video_files


def run_openface_on_video(video_path, openface_dir, output_folder):
    """
    Chạy OpenFace để trích xuất đặc trưng từ video và lưu kết quả vào file CSV.
    Args:
    - video_path (str): Đường dẫn đến video cần xử lý.
    - openface_dir (str): Thư mục chứa OpenFace (đường dẫn đến OpenFace).
    - output_folder (str): Thư mục lưu kết quả output (file CSV).

    Returns:
    - str: Đường dẫn đến file CSV đầu ra.
    """
    output_file = os.path.join(output_folder, os.path.basename(video_path).replace('.avi', '.csv'))
    command = [
        os.path.join(openface_dir, 'bin', 'FeatureExtraction'),
        '-f', video_path,
        '-out_dir', output_folder
    ]

    # Chạy lệnh OpenFace
    try:
        subprocess.run(command, check=True)
        print(f"Đã trích xuất đặc trưng từ video {video_path}. Kết quả lưu tại {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi xử lý video {video_path}: {e}")
        return None


def process_video_files(input_folder, openface_dir, output_folder):
    """
    Tiến hành xử lý tất cả video trong thư mục đầu vào và trích xuất các đặc trưng với OpenFace.
    Args:
    - input_folder (str): Thư mục chứa video.
    - openface_dir (str): Đường dẫn đến thư mục OpenFace.
    - output_folder (str): Thư mục lưu các file CSV đầu ra.
    """
    video_files = find_video_files(input_folder)

    if not video_files:
        print("Không tìm thấy video nào trong thư mục.")
        return

    for video_file in video_files:
        run_openface_on_video(video_file, openface_dir, output_folder)


if __name__ == "__main__":
    # Đường dẫn đến thư mục Train trong bộ dữ liệu DAiSEE
    input_folder = "/media/binh/New Volume/Binh/data/DAiSEE/DAiSEE/DataSet/Train"
    # Đường dẫn đến thư mục OpenFace
    openface_dir = "/home/binh/Workspace/OpenFace/build"
    # Đường dẫn đến thư mục lưu trữ kết quả
    output_folder = "/media/binh/New Volume/Binh/data/DAiSEE/DAiSEE/Features"

    # Kiểm tra và tạo thư mục lưu kết quả nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Thực hiện quá trình xử lý
    process_video_files(input_folder, openface_dir, output_folder)
