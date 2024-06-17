import os
import csv
# import path
import pandas as pd
import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt

Bearing_label = ["N","SB","SI","SO"]
raw_path = "C:\\Users\\user\\Desktop\\연구실\\데이터\\CASE_WESTERN_RESERVE\\transfer_DE\\0.007"

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
        class_data = []
        for i in range(len(DATA[j])):
            arr = DATA[j][i].to_numpy()
            if len(arr) >= 1200:
                for k in range(0, len(arr)-1200+1, 1200):
                    x = arr[k:k+1200].reshape(1, 1200)
                    class_data.append(x)
            else:
                continue
        Total_data.append(np.vstack(class_data) if class_data else None)
    return Total_data

arr_data = DATA_ADD(DATA)

#윈도우 크기(64*64)
window_size = 126
overlap = 107

folders = ["N","SB","SI","SO"]
save_folder = 'C:\\Users\\user\\Desktop\\연구실\\데이터\\CASE_WESTERN_RESERVE\\transfer_DE_spectrogram시각화'

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
#         save_filename = os.path.join(save_folder, folder_path, f"spectrogram_{i}.npz")
#         class_label = j
#         # class_labels = np.full_like(spectrogram, class_label,dtype=np.int64)
#         # np.savez(save_filename,x=spectrogram, y=class_labels)
#         np.savez(save_filename,x=spectrogram, y=class_label)
#         print(f"Saved spectrogram for {folder_path}/{i}")

import scipy.ndimage


def resize_spectrogram(spectrogram, target_shape=(64, 64)):
    return scipy.ndimage.zoom(spectrogram,
                              (target_shape[0] / spectrogram.shape[0], target_shape[1] / spectrogram.shape[1]))

# for j in range(len(arr_data)):
#     # 폴더 경로
#     folder_path = folders[j]
#     arr_data_list = arr_data[j]
#
#     for i in range(len(arr_data_list)):
#         # STFT 계산
#         frequencies, times, spectrogram = signal.stft(arr_data_list[i], nperseg=window_size, noverlap=overlap)
#
#         # 절대값을 취해서 크기를 맞추기
#         spectrogram = np.abs(spectrogram)
#         resized_spectrogram = resize_spectrogram(spectrogram, target_shape=(64, 64))
#         # 스펙트로그램 저장
#         save_filename = os.path.join(save_folder,folder_path, f"2024_{folder_path}_spectrogram_{i}.npz")
#         class_label = j
#         np.savez(save_filename, x=resized_spectrogram, y=class_label)
#         print(f"Saved spectrogram for {folder_path}/{i}")

##################################################################################

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

print(save_folder)