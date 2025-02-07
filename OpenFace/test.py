import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến file CSV đã merge
merged_csv_path = "/home/binh/Workspace/data/data_science/data_finally/output_merged.csv"  # Thay bằng đường dẫn thật

# Đọc file CSV
merged_df = pd.read_csv(merged_csv_path)

# Kiểm tra xem cột "engagement" có tồn tại không
if 'Engagement' not in merged_df.columns:
    print("Cột 'engagement' không tồn tại trong file CSV.")
else:
    # In phân phối cơ bản (giá trị đếm)
    print("Phân phối của cột engagement:")
    print(merged_df['Engagement'].value_counts())

    # Vẽ biểu đồ phân phối
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df['Engagement'], bins=20, kde=True, color='blue')
    plt.title("Phân phối của cột Engagement", fontsize=16)
    plt.xlabel("Giá trị Engagement", fontsize=12)
    plt.ylabel("Tần suất", fontsize=12)
    plt.grid(axis='y')
    plt.show()
