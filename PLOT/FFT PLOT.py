###############################frequency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 경로 (로컬 경로로 변경)
csv_file_path = "C:\\Users\\user\\Desktop\\연구실\\데이터\\data\\2024\\6207\\900\\7_SO\\7_SO_2403201943.csv"
# 데이터 읽어오기
df = pd.read_csv(csv_file_path)
# 주파수 스펙트럼 계산
data_per_second = 1024
seconds = 50
data_chunks = []

# 전체 데이터를 초당 데이터로 분할
for i in range(0, len(df), data_per_second):
    data_chunk = df['vibration'][i:i+data_per_second].values
    data_chunks.append(data_chunk)

# FFT 및 주파수 스펙트럼 계산
sample_rate = 8192  # 샘플링 속도
freq = np.fft.fftfreq(data_per_second, 1 / sample_rate)  # 주파수 계산

for second, data_chunk in enumerate(data_chunks):
    fft_result = np.fft.fft(data_chunk)
    magnitude = np.abs(fft_result)

    # 주파수가 양수인 부분을 그래프로 표시
    positive_freq = freq[:data_per_second // 2]
    magnitude = magnitude[:data_per_second // 2]

    plt.figure(dpi=100, figsize=(15, 5))
    plt.plot(positive_freq, magnitude, color='royalblue',linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f' {second+1}')
    plt.tight_layout()

    # 이미지 파일로 저장
    output_file_path = f"C:/Users/user/Desktop/연구실/시각화/FFT/2024_{second+1}_second_plot.png"
    plt.savefig(output_file_path, format='png')
    plt.close()  # 그래프 창 닫기
    print(f"{output_file_path}에 주파수 스펙트럼 저장 완료")


###################################
    import csv
    import matplotlib.pyplot as plt
    import numpy as np

    # CSV 파일 경로
    csv_file_path = r"C:\Users\gyuhee\Desktop\data\bearing\6208_clean\8_N_2\8_N_2_2208091917.csv"

    # 데이터 수집 속도
    data_per_second = 10240

    # 초 단위 데이터 저장
    data_chunks = []

    # csv 파일 불러오기
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        total_data = []
        for row in csv_reader:
            total_data.append(float(row['vibration']))

        for i in range(0, len(total_data), data_per_second):
            data_chunk = total_data[i:i + data_per_second]
            data_chunks.append(data_chunk)

    # 주파수 영역 그래프를 그리고 이미지 파일로 저장
    for second, data_chunk in enumerate(data_chunks):
        plt.rc('font', family='NanumBarunGothic')
        plt.figure(dpi=100, figsize=(5, 3))

        # 주파수 영역 계산
        N = len(data_chunk)
        T = 1.0 / data_per_second
        x = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        yf = np.fft.fft(data_chunk)
        xf = 1.0 / (2.0 * T) * np.abs(yf[:N // 2])

        plt.plot(x, 2.0 / N * np.abs(yf[:N // 2]), linewidth=3, color='green')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'{second + 1}초의 주파수 영역 데이터')
        plt.tight_layout()

        # 이미지 파일로 저장
        output_file_path = f"C:\\Users\\gyuhee\\Desktop\\베어링_png\\plot\\DE\\{second + 1}_second_plot.png"
        plt.savefig(output_file_path, format='png')
        plt.close()  # 그래프 창 닫기
        print(f"{output_file_path}에 그래프 저장 완료")

