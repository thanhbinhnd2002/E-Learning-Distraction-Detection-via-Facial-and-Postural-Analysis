import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Đường dẫn đến thư mục chứa file CSV đầu ra từ OpenFace
input_folder = "/home/binh/Workspace/projects/OpenFace/build/bin/output/"
output_csv = "/home/binh/Workspace/projects/OpenFace/build/bin/processed_features.csv"

# Lấy danh sách tất cả các file CSV
def get_csv_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

all_csv_files = get_csv_files(input_folder)
print(f"Tìm thấy {len(all_csv_files)} file CSV để tiền xử lý.")

# Kiểm tra nếu không có file CSV nào
if not all_csv_files:
    print("Không có file CSV nào để tiền xử lý.")
    exit()

# Danh sách lưu trữ kết quả từ tất cả các file CSV
all_data = []

# Các cột đặc trưng cần thiết
columns_to_keep = ['frame', 'timestamp', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'AU06_r', 'AU12_r', 'AU45_r']

# Đọc và tiền xử lý từng file CSV
for csv_file in all_csv_files:
    try:
        # Đọc file CSV
        df = pd.read_csv(csv_file)
        video_name = os.path.basename(csv_file).split('.')[0]
        print(f"Đang xử lý file CSV: {video_name}")

        # Lọc các cột cần thiết
        if not all(col in df.columns for col in columns_to_keep):
            print(f"Bỏ qua file {csv_file} do thiếu các cột cần thiết.")
            continue
        df = df[columns_to_keep]

        # Nội suy giá trị thiếu
        df.interpolate(method='linear', inplace=True)

        # Loại bỏ ngoại lệ bằng Isolation Forest
        features = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'AU06_r', 'AU12_r', 'AU45_r']
        clf = IsolationForest(contamination=0.01, random_state=42)
        df['is_outlier'] = clf.fit_predict(df[features])
        df = df[df['is_outlier'] == 1]
        df.drop(columns=['is_outlier'], inplace=True)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

        # Chia dữ liệu theo cửa sổ thời gian và tạo các đặc trưng thống kê
        WINDOW_SIZE = 300  # Tương ứng với 10 giây cho video 30 fps
        processed_data = []

        for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = df.iloc[i:i + WINDOW_SIZE][features]

            # Tính các đặc trưng thống kê
            mean_features = window.mean().values
            std_features = window.std().values
            max_features = window.max().values
            min_features = window.min().values

            # Gộp các đặc trưng
            combined_features = np.hstack([mean_features, std_features, max_features, min_features])
            processed_data.append(combined_features)

        # Tạo DataFrame cho từng video và lưu lại
        columns = [f'{stat}_{feat}' for stat in ['mean', 'std', 'max', 'min'] for feat in features]
        video_df = pd.DataFrame(processed_data, columns=columns)
        video_df['video_name'] = video_name
        all_data.append(video_df)

    except Exception as e:
        print(f"Lỗi khi xử lý file {csv_file}: {e}")
        continue

# Gộp tất cả dữ liệu lại và lưu vào file CSV cuối cùng
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Quá trình tiền xử lý hoàn tất. Dữ liệu đã được lưu tại {output_csv}.")
else:
    print("Không có dữ liệu để lưu.")
