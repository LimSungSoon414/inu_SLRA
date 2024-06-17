import os
import numpy as np

loaded_x = np.load('C:/Users/user/Documents/카카오톡 받은 파일/Inner_12_DE_021_209_0_DE..csv_chunk20.npy')

print(loaded_x)


# NPY 데이터 주소
base_folder = "C:/Users/user/Desktop/연구실/NPY/similarity"
folders = ['N', 'SI', 'SO', 'SB']

# 저장할 CSV 주소의 기본 경로
output_base_folder = "C:/Users/user/Desktop/연구실/데이터/Eated_data/6207"

for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    output_folder_path = os.path.join(output_base_folder, folder)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)

            data = np.load(file_path)
            csv_file_name = file_name.replace('.npy', '.csv')
            csv_file_path = os.path.join(output_folder_path, csv_file_name)

            np.savetxt(csv_file_path, data, delimiter=',', fmt='%f', header='vibration', comments='')

            print(f"Saved {csv_file_path}")

print("All files saved as CSV!")