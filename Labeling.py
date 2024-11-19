import pandas as pd

# Đường dẫn tới file CSV đã chuẩn hóa
file_path = "/home/binh/Workspace/projects/OpenFace/build/bin/processed_features.csv"
output_path = "/home/binh/Workspace/projects/OpenFace/build/bin/labeled_features.csv"

# Đọc file CSV đã chuẩn hóa
df = pd.read_csv(file_path)

# Các cột đặc trưng cần chuẩn hóa ngược lại
features = ['mean_pose_Rx', 'mean_pose_Ry', 'mean_pose_Rz', 'mean_AU06_r', 'mean_AU12_r', 'mean_AU45_r']
std_features = ['std_pose_Rx', 'std_pose_Ry', 'std_pose_Rz', 'std_AU06_r', 'std_AU12_r', 'std_AU45_r']
mean_features = ['mean_pose_Rx', 'mean_pose_Ry', 'mean_pose_Rz', 'mean_AU06_r', 'mean_AU12_r', 'mean_AU45_r']

# Trung bình và độ lệch chuẩn từ bước chuẩn hóa
# Lưu ý: Những giá trị này cần được lưu từ quá trình chuẩn hóa ban đầu
mean_values = {
    'mean_pose_Rx': 0, 'mean_pose_Ry': 0, 'mean_pose_Rz': 0,
    'mean_AU06_r': 0, 'mean_AU12_r': 0, 'mean_AU45_r': 0
}
std_values = {
    'mean_pose_Rx': 1, 'mean_pose_Ry': 1, 'mean_pose_Rz': 1,
    'mean_AU06_r': 1, 'mean_AU12_r': 1, 'mean_AU45_r': 1
}

# Chuẩn hóa ngược lại
for feature in features:
    df[feature] = df[feature] * std_values[feature] + mean_values[feature]

# Gán nhãn tập trung mặc định
df['label'] = 0

# Gán nhãn mất tập trung dựa trên giá trị gốc
df.loc[(df['mean_pose_Ry'] > 20) | (df['mean_pose_Ry'] < -20), 'label'] = 1
df.loc[(df['mean_pose_Rx'] > 15) | (df['mean_pose_Rx'] < -15), 'label'] = 1
df.loc[df['mean_AU45_r'] > 0.5, 'label'] = 1

# Lưu file CSV đã gán nhãn
df.to_csv(output_path, index=False)
print(f"Đã lưu dữ liệu đã gán nhãn tại: {output_path}")
