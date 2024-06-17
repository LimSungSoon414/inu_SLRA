import os
import csv
import pandas as pd
import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt


Bearing_label = ["KA04"]
raw_path = "C:\\Users\\user\\Desktop\\연구실\\데이터\\Paderborn_BearingDATASET\\RORD\\"


def CSV_READER(Bearing_label, raw_path):

    Total_DATA = []

    for i in Bearing_label:
        print('현재_' + str(i) + '_진행증')
        directory = os.path.join(raw_path,i)
        mid_data = []

        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                data = pd.read_csv(file_path)
                # [1:2] vibration ,  [2:3] Acoustic
                mid_data.append(data.iloc[:,1:2])

        Total_DATA.append(mid_data)

    return Total_DATA

DATA = CSV_READER(Bearing_label, raw_path)

def DATA_ADD(DATA):
    Total_data = []

    for j in range(len(DATA)):
        save = []
        for i in range(len(DATA[j])):
            arr = DATA[j][i].to_numpy()
            if len(arr) == int(80*6400):
                x = arr.reshape(80, 6400)
            else:
                x = arr[:512000].reshape(80, -1)
            save.append(x)
        Total_data.append(np.concatenate(save))
    return Total_data

arr_data = DATA_ADD(DATA)

# 윈도우 크기(128*128)
# window_size = 255
# overlap = 174

# #윈도우 크기(64*64)
window_size = 127
overlap = 25

folders = ["KA04"]
save_folder = 'C:\\Users\\user\\Desktop\\연구실\\시각화\\PU_BearingData_spectrogram\\RORD'

# for j in range(len(arr_data)):
#     # 폴더 경로
#     folder_path = folders[j]
#     arr_data_list = arr_data[j]
#
#     for i in range(len(arr_data_list)):
#
#         # STFT 계산
#         frequencies, times, spectrogram = signal.stft(arr_data_list[i], nperseg=window_size, noverlap=overlap)
#         # STFT.append(spectrogram)
#
#         save_filename = os.path.join(save_folder, folder_path, f"2023_{folder_path}_spectrogram_{i}.npz")
#         class_label = j
#         # class_labels = np.full_like(spectrogram, class_label,dtype=np.int64)
#         # np.savez(save_filename,x=spectrogram, y=class_labels)
#         np.savez(save_filename,x=spectrogram, y=class_label)
#         print(f"Saved spectrogram for {folder_path}/{i}")

########################################################################################################################

for j in range(len(arr_data)):
    # 폴더 경로
    folder_path = folders[j]
    arr_data_list = arr_data[j]

    for i in range(len(arr_data_list)):

        # STFT 계산
        frequencies, times, spectrogram = signal.stft(arr_data_list[i], nperseg=window_size, noverlap=overlap)
        # STFT.append(spectrogram)

        # 스펙트로그램 시각화
        plt.figure()
        plt.pcolormesh(times, frequencies, np.abs(spectrogram), shading='auto')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'{folder_path} Spectrogram - {i+1}')
        # plt.show()

        # 파일 저장
        filename = f'{folder_path}_spectrogram_{i + 1}.png'
        save_path = os.path.join(save_folder,folder_path, filename)
        plt.savefig(save_path)
        plt.close()

# print(save_folder)