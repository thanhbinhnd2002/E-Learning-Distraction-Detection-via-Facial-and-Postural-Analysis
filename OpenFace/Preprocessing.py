import os

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def get_csv_files(folder_path):
    """
    Tìm tất cả các tệp CSV trong thư mục đầu vào.
    Args:
        folder_path (str): Đường dẫn thư mục chứa các file CSV.
    Returns:
        list: Danh sách các đường dẫn file CSV.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.lower().endswith('.csv')
    ]

def preprocess_csv(file_path, columns_to_keep, contamination=0.01):
    """
    Tiền xử lý một file CSV: lọc cột, nội suy giá trị thiếu, loại bỏ ngoại lệ, và chuẩn hóa.
    Args:
        file_path (str): Đường dẫn tới file CSV.
        columns_to_keep (list): Danh sách các cột cần giữ lại.
        contamination (float): Tỷ lệ ngoại lệ (IsolationForest).
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu đã tiền xử lý.
    """
    try:
        df = pd.read_csv(file_path)

        # Kiểm tra các cột yêu cầu
        if not all(col in df.columns for col in columns_to_keep):
            print(f"Bỏ qua file {file_path} do thiếu cột cần thiết.")
            return None

        # Lọc các cột cần thiết
        df = df[columns_to_keep]

        # Nội suy giá trị thiếu
        df.interpolate(method='linear', inplace=True)

        # Loại bỏ ngoại lệ
        features = columns_to_keep[2:]  # Giả định bỏ cột frame, timestamp
        clf = IsolationForest(contamination=contamination, random_state=42)
        df['is_outlier'] = clf.fit_predict(df[features])
        df = df[df['is_outlier'] == 1].drop(columns=['is_outlier'])

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

        # Lưu scaler
        joblib.dump(scaler, "checkpoints/scaler.pkl")
        print("Scaler đã được lưu tại checkpoints/scaler.pkl")

        return df

    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

def extract_mean_features(df, video_name, window_size=300):
    """
    Tính toán đặc trưng trung bình (mean) từ một cửa sổ thời gian và thêm cột tên video.
    Args:
        df (pd.DataFrame): DataFrame đã tiền xử lý.
        video_name (str): Tên video tương ứng.
        window_size (int): Kích thước cửa sổ (số khung hình).
    Returns:
        pd.DataFrame: DataFrame chứa đặc trưng trung bình (mean) và tên video.
    """
    features = df.columns[2:]  # Giả định bỏ cột frame, timestamp
    processed_data = []

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size][features]
        mean_features = window.mean().values
        processed_data.append(mean_features)

    result_df = pd.DataFrame(processed_data, columns=[f"mean_{feat}" for feat in features])
    result_df["video_name"] = video_name  # Thêm tên video vào mỗi dòng
    return result_df

def save_processed_data(dataframes, output_file):
    """
    Ghi tất cả các DataFrame đã xử lý vào một file CSV duy nhất.
    Args:
        dataframes (list): Danh sách DataFrame cần ghi.
        output_file (str): Đường dẫn file CSV đầu ra.
    """
    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Dữ liệu đã được lưu tại {output_file}.")
    else:
        print("Không có dữ liệu nào để lưu.")

def preprocess_all(input_folder, output_file, columns_to_keep, window_size=300):
    """
    Tiền xử lý tất cả các file CSV trong thư mục đầu vào và lưu kết quả.
    Args:
        input_folder (str): Thư mục chứa file CSV.
        output_file (str): Đường dẫn file CSV đầu ra.
        columns_to_keep (list): Danh sách các cột cần giữ lại.
        window_size (int): Kích thước cửa sổ thời gian.
    """
    csv_files = get_csv_files(input_folder)
    print(f"Tìm thấy {len(csv_files)} file CSV để tiền xử lý.")

    if not csv_files:
        print("Không có file CSV nào để tiền xử lý.")
        return

    all_data = []
    for csv_file in csv_files:
        video_name = os.path.basename(csv_file).split(".")[0]  # Lấy tên file không bao gồm phần mở rộng
        print(f"Đang xử lý file: {csv_file} (Video: {video_name})")
        preprocessed_df = preprocess_csv(csv_file, columns_to_keep)
        if preprocessed_df is not None:
            mean_features_df = extract_mean_features(preprocessed_df, video_name, window_size)
            all_data.append(mean_features_df)

    save_processed_data(all_data, output_file)

if __name__ == "__main__":
    input_folder = "/home/binh/Output"
    output_file = "/home/binh/Output/test/test.csv"
    columns_to_keep = ['frame', 'timestamp', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'AU06_r', 'AU12_r', 'AU45_r']
    preprocess_all(input_folder, output_file, columns_to_keep)
