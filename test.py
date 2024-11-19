import pandas as pd
import sys

video_path = "/home/binh/Workspace/projects/OpenFace/build/bin/output/video10_mp4/video10.csv"

df = pd.read_csv(video_path)
columns_to_keep = ['frame', 'timestamp', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'AU06_r', 'AU12_r', 'AU45_r']
df = df[columns_to_keep]
print(df.head)
