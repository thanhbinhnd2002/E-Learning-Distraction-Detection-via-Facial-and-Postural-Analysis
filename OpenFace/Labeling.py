import pandas as pd

def label_driver(input_csv, output_csv):
    # Đọc file CSV
    df = pd.read_csv(input_csv)

    # Kiểm tra nếu cột 'video_name' tồn tại
    if 'video_name' in df.columns:
        # Kiểm tra cột 'video_name' và đánh nhãn
        df['label_1'] = df['video_name'].apply(lambda x: 1 if 'driver' in str(x).lower() else 0)
    else:
        print("Cột 'video_name' không tồn tại trong file CSV.")
        return

    # Lưu lại file CSV đã đánh nhãn, chỉ giữ các cột cần thiết
    df = df[['video_name', 'label_1'],]  # Chỉ giữ lại cột 'video_name' và 'label_1'
    df.to_csv(output_csv, index=False)

    print(f"File đã được đánh nhãn và lưu tại {output_csv}")

# Sử dụng hàm
# Thay 'input.csv' bằng đường dẫn file CSV đầu vào và 'output.csv' là file đầu ra
input_file = '/home/binh/Workspace/data/data_science/data_finally/data_finally.csv'
output_file = '/home/binh/Workspace/data/data_science/test/test.csv'
label_driver(input_file, output_file)
