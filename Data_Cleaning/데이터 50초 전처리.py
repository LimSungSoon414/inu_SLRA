import os
import glob
import pandas as pd
import numpy as np
# Raw data를 간단한 전처리를 통해 data 폴더에 저장

def Convert_Raw_Data(raw_path, save_path, cut_point, total_size, unit=10240, statistics=False):
    # # path setting
    # raw_path = os.path.join(root_dir, "raw_data", folder_name)
    # data_path = os.path.join(root_dir, "data_50sec", folder_name)

    # load raw data
    raw_list = Trim_csv(path=raw_path, cut_point=cut_point, total_size=total_size, unit=unit)

    # make directory
    os.makedirs(save_path, exist_ok=True)

    # Save Trimed data
    Save_list_csv(raw_list, raw_path, save_path, prefix=save_path.split('\\')[-1])

    return print(f"done")

# path에서 num_row만큼 csv를 불러오면서 간단한 전처리를 실시
def Trim_csv(path, cut_point, total_size, unit):
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

    df_list = []

    for csv in csv_files:
        df_list.append(pd.read_csv(filepath_or_buffer=path +'\\'+ csv, encoding='cp949',
                                   skiprows=unit*cut_point, nrows=unit*total_size, sep='\t', usecols=[1, 2],
                                   header=0, names=['vibration', 'acoustic']))
    return df_list

# df_list를 path에 .csv로 저장
def Save_list_csv(df_list, raw_path, save_path, prefix):
    csv_id = [f for f in os.listdir(raw_path) if f.endswith(".csv")]

    for df_idx ,df in enumerate(df_list):
        df.to_csv(save_path + '\\'+ prefix + f"_{csv_id[df_idx][-14:-4]}.csv")
    return

# Convert_Raw_Data('7_N_3', 512000, 'C:/Users/user/Desktop/연구실/데이터')

# PATH = os.getcwd()
PATH = 'C:/Users/user/Desktop/연구실/데이터'

raw_path = os.path.join(PATH, 'raw_data/2023/6208', '8_WI')
save_path = os.path.join(PATH, 'data/2023/6208', '8_WI')

Convert_Raw_Data(raw_path=raw_path, save_path=save_path, cut_point=10, total_size=50,
                 unit=10240, statistics=False)