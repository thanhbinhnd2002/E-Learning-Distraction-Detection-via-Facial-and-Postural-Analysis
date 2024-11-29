import pandas as pd
import numpy as np

# Đọc file CSV
file_path = "/home/binh/Output/Final_Data/processed_features.csv"  # Thay bằng đường dẫn file CSV của bạn
df = pd.read_csv(file_path)

# Tạo cột Label với giá trị ngẫu nhiên (0 hoặc 1)
np.random.seed(42)  # Đặt seed để đảm bảo kết quả tái hiện được
df['Label'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])  # Tỷ lệ 50:50

# Kiểm tra tỷ lệ nhãn
label_counts = df['Label'].value_counts(normalize=True)
print("Tỷ lệ nhãn (0 và 1):")
print(label_counts)

# Lưu file mới
output_path = "labeled_file_random.csv"
df.to_csv(output_path, index=False)
print(f"File đã được lưu với nhãn tại: {output_path}")
