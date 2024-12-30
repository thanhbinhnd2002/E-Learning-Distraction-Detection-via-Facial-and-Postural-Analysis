import pandas as pd

# Đọc dữ liệu
data = pd.read_csv("processed/output_features.csv")

# Chọn các cột cần thiết
raw_columns_to_keep = ['gaze_angle_x', 'gaze_angle_y', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                       'pose_Tx', 'pose_Ty', 'pose_Tz', 'AU06_r', 'AU45_r']
raw_features = data[raw_columns_to_keep]

# Tính toán các đặc trưng trung bình
window_size = 150  # 5 giây với 30 FPS
processed_data = []

for i in range(0, len(raw_features) - window_size + 1, window_size):
    window = raw_features.iloc[i:i + window_size]
    mean_features = window.mean(axis=0)

    # Append giá trị của mean_features dưới dạng danh sách
    processed_data.append(mean_features.tolist())

# Đặt tên cột theo format mong muốn
columns = [f"mean_{feat}" for feat in raw_columns_to_keep]

# Tạo DataFrame từ các đặc trưng đã tính toán
processed_df = pd.DataFrame(processed_data, columns=columns)

# Hiển thị kết quả
print(processed_df)
