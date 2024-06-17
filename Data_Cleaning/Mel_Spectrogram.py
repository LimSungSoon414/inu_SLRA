import os
import csv
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


Bearing_label = ["7_N","7_SB","7_SI","7_SO","7_WB","7_WI","7_WO"]
raw_path = "C:\\Users\\user\\Desktop\\연구실\\데이터\\data\\2024\\6207\\300"


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
                mid_data.append(data.iloc[:,2:3])

        Total_DATA.append(mid_data)

    return Total_DATA

def DATA_ADD(DATA):
    Total_data = []

    for j in range(len(DATA)):
        # print(j)
        save = []
        for i in range(len(DATA[j])):
            DATA[j][i]
            arr = DATA[j][i].to_numpy()
            if len(arr) == int(50*8192):
                x = arr.reshape(50, 8192)
                save.append(x)
            else:
                continue
        Total_data.append(np.concatenate(save))
    return Total_data

DATA = CSV_READER(Bearing_label, raw_path)
arr_data = DATA_ADD(DATA)

# Parameters for STFT
n_fft = 512
hop_length = 129
n_mels = 64

folders = ["N", "SB", "SI", "SO", "WB", "WI", "WO"]
save_folder = 'C:\\Users\\user\\Desktop\\연구실\\spectrogram\\Mel_spectrogram\\300'

# for j in range(len(arr_data)):
#     folder_path = folders[j]
#     arr_data_list = arr_data[j]
#
#     for i in range(len(arr_data_list)):
#         # STFT 계산
#         mel_spectrogram = librosa.feature.melspectrogram(y=arr_data_list[i].flatten(), sr=8192, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#         mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
#         # Save the spectrogram as npz file
#         save_filename = os.path.join(save_folder, folder_path, f"2024_{folder_path}_Melspectrogram_{i}.npz")
#         class_label = j
#         np.savez(save_filename, x=mel_spectrogram_db, y=class_label)
#         print(f"Saved spectrogram for {folder_path}/{i}")

for j in range(len(arr_data)):
    folder_path = folders[j]
    arr_data_list = arr_data[j]

    for i in range(len(arr_data_list)):
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=arr_data_list[i].flatten(), sr=8192, n_fft=n_fft,
                                                         hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Plot Mel spectrogram
        plt.figure()
        librosa.display.specshow(mel_spectrogram_db, sr=8192, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{folder_path} Mel Spectrogram - {i + 1}')
        plt.tight_layout()

        # Save the Mel spectrogram as image file
        filename = f'{folder_path}_melspectrogram_{i + 1}.png'
        save_path = os.path.join(save_folder, folder_path, filename)
        plt.savefig(save_path)
        plt.close()

