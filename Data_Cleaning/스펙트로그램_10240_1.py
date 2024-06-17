import os
import csv
import pandas as pd
import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt


Bearing_label = ["N","SB","SI","SO"]
raw_path = "C:/Users/user/Desktop/연구실/데이터/Eated_data/6207"


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
                mid_data.append(data.iloc[:])

        Total_DATA.append(mid_data)

    return Total_DATA

DATA = CSV_READER(Bearing_label, raw_path)

def concatenate_data(DATA):
    concatenated_data = []

    for j in range(len(DATA)):
        save = []
        for i in range(len(DATA[j])):
            arr = DATA[j][i].to_numpy()
            if arr.shape == (10240, 1):
                save.append(arr)
            else:
                continue
        if save:
            concatenated_data.append(np.concatenate(save, axis=1))
        else:
            concatenated_data.append(None)
    return concatenated_data

arr_data = concatenate_data(DATA)

def transpose_data(data_list):
    transposed_data = []
    for data in data_list:
        if data is not None:
            transposed_data.append(data.T)
        else:
            transposed_data.append(None)
    return transposed_data

transposed_arr_data = transpose_data(arr_data)
# 윈도우 크기(128*128)
# window_size = 255
# overlap = 174

#윈도우 크기(64*64)
window_size = 127
overlap = -36

folders = ["N","SB","SI","SO"]
save_folder = 'C:/Users/user/Desktop/연구실/데이터/Eated_data/SPECTROGRAM/6207/시각화'

for j in range(len(transposed_arr_data)):
    # 폴더 경로
    folder_path = folders[j]
    arr_data_list = transposed_arr_data[j]

    for i in range(len(arr_data_list)):

        # STFT 계산
        frequencies, times, spectrogram = signal.stft(arr_data_list[i], nperseg=window_size, noverlap=overlap)
        # STFT.append(spectrogram)

        save_filename = os.path.join(save_folder, folder_path, f"spectrogram_{i}.npz")
        class_label = j
        # class_labels = np.full_like(spectrogram, class_label,dtype=np.int64)
        # np.savez(save_filename,x=spectrogram, y=class_labels)
        np.savez(save_filename,x=spectrogram, y=class_label)
        print(f"Saved spectrogram for {folder_path}/{i}")


########################################################################################################################

# for j in range(len(transposed_arr_data)):
#     # 폴더 경로
#     folder_path = folders[j]
#     arr_data_list = transposed_arr_data[j]
#
#     for i in range(len(arr_data_list)):
#
#         # STFT 계산
#         frequencies, times, spectrogram = signal.stft(arr_data_list[i], nperseg=window_size, noverlap=overlap)
#         # STFT.append(spectrogram)
#
#         # 스펙트로그램 시각화
#         plt.figure()
#         plt.pcolormesh(times, frequencies, np.abs(spectrogram), shading='auto')
#         plt.colorbar(label='Amplitude')
#         plt.xlabel('Time')
#         plt.ylabel('Frequency')
#         plt.title(f'{folder_path} Spectrogram - {i+1}')
#         # plt.show()
#
#         # 파일 저장
#         filename = f'{folder_path}_spectrogram_{i + 1}.png'
#         save_path = os.path.join(save_folder,folder_path, filename)
#         plt.savefig(save_path)
#         plt.close()

# print(save_folder)