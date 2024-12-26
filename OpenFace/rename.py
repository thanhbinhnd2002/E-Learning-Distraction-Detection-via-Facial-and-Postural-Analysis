import os
import shutil

def rename_videos(input_folders):
    """
    Đổi tên các video trong thư mục theo dạng <tên_thư_mục_cha>_video_{i}, đánh số thứ tự trong mỗi thư mục.
    Args:
        input_folders (list): Danh sách các thư mục chứa video.
    """
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            parent_folder = os.path.basename(root)
            files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.csv'))]
            files.sort()  # Đảm bảo thứ tự nhất quán

            index = 1
            for file in files:
                file_path = os.path.join(root, file)
                new_name = f"{parent_folder}_video_{index}{os.path.splitext(file)[1]}"
                new_path = os.path.join(root, new_name)

                try:
                    shutil.move(file_path, new_path)
                    print(f"Đã đổi tên: {file_path} -> {new_path}")
                    index += 1
                except Exception as e:
                    print(f"Lỗi khi đổi tên file {file_path}: {e}")

if __name__ == "__main__":
    input_folders = [
        "/home/binh/Workspace/data/data_science/data_raw_1"
    ]

    rename_videos(input_folders)
    print("Quá trình đổi tên hoàn tất.")
