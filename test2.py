import pandas as pd
file_path = "/home/binh/Workspace/projects/OpenFace/build/bin/output/video2_mp4/video2.csv"
df = pd.read_csv(file_path, delimiter=';', engine='python')
print(df.head())