import os
import csv
import numpy as np

folder_path = r"C:\Users\user\Desktop\연구실\데이터\data\2023\6208"
# data_per_second = 400
# seconds = 300
data_per_second = 10240
seconds = 50

# 클래스 및 폴더 목록
# classes = ['7_N_3', '7_SB_1', '7_SI_1', '7_SO_1', '7_WB_1', '7_WI_1', '7_WO_1']
# classes = ['N', 'SB', 'SI', 'SO','WB', 'WI', 'WO']
classes = ['8_SI_1','8_SO_1']


for class_name in classes:
    class_folder_path = os.path.join(folder_path, class_name)
    csv_files = os.listdir(class_folder_path)

    for csv_file_name in csv_files:
        csv_file_path = os.path.join(class_folder_path, csv_file_name)

        data_chunks = []

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            total_data = []
            for row in csv_reader:
                total_data.append(float(row['vibration']))

            # 1초씩 데이터를 끊어서 NumPy 배열로 저장
            data_chunk_size = data_per_second
            num_chunks = len(total_data) // data_chunk_size
            for i in range(num_chunks):
                start_idx = i * data_chunk_size
                end_idx = (i + 1) * data_chunk_size
                data_chunk = np.array(total_data[start_idx:end_idx])
                data_chunks.append(data_chunk)

        for second_idx, data_chunk in enumerate(data_chunks):
            # save_folder = os.path.join(folder_path, f"{class_name}_numpy")
            save_folder = os.path.join(folder_path, f"{class_name}_numpy")
            os.makedirs(save_folder, exist_ok=True)
            save_file_path = os.path.join(save_folder, f"{csv_file_name}_second{second_idx + 1}.npy")

            # 데이터 저장
            np.save(save_file_path, data_chunk)
            print(f"Saved data chunk {second_idx + 1} to {save_file_path}")
