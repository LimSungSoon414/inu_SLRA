import csv
import matplotlib.pyplot as plt

# csv 파일 경로 (로컬 경로로 변경)
csv_file_path = r"C:\Users\user\Desktop\연구실\데이터\data\2024\6208\900\8_N\8_N_2402281813.csv"


data_per_second = 1024

seconds = 50
data_chunks = []

# csv 파일 불러오기
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    total_data = []
    for row in csv_reader:
        #vibration, acoustic
        total_data.append(float(row['acoustic']))

    for i in range(0, len(total_data), data_per_second):
        data_chunk = total_data[i:i+data_per_second]
        data_chunks.append(data_chunk)

# 초별 데이터를 그래프로 그리고 이미지 파일로 저장
for second, data_chunk in enumerate(data_chunks):
# for second, data_chunk in range(10):
    plt.rc('font', family='NanumBarunGothic')
    plt.figure(dpi=100,linewidth=0.1,figsize=(25, 5))
    plt.plot(data_chunk, color='royalblue',linewidth = "2.5")
    plt.xlabel('time')
    plt.ylabel('vibration')
    plt.title(f'07_N_900_Vibration_data')
    plt.tight_layout()

    # 이미지 파일로 저장
    output_file_path = f"C:\\Users\\user\\Desktop\\연구실\\시각화\\스펙트로그램_시각화\\2024_version\\6208\\N\\1_{second+1}_second_plot.png"
    plt.savefig(output_file_path, format='png')
    plt.close()  # 그래프 창 닫기
    print(f"{output_file_path}에 그래프 저장 완료")


#################
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한국어 폰트 설정
font_path = "C:\\Windows\\Fonts\\malgun.ttf"
font_prop = fm.FontProperties(fname=font_path, size=15)

data_size = [10, 30, 50, 70]

# Model_A = [28.21, 36.91, 43.14, 59.69]
# Model_B = [35.83, 42.36, 46.17, 47.02]
# Model_C = [72.44, 73.03, 74.96, 75.69]
# # Model_D = [86.13, 84.54, 83.81, 87.79]
Model_A = [26.43, 31.06, 33.61, 38.67]
Model_B = [35.83, 42.36, 46.17, 47.02]
Model_C = [78.47, 72.66, 72.63, 76.72]
Model_D = [66.37, 81.69, 83.81, 87.79]
# Model_A = [60.00, 70.00, 72.50, 74.64]
# Model_B = [69.00, 72.67, 76.80, 77.86]
# Model_C = [70.42, 72.92, 73.33, 75.89]
# Model_D = [71.25, 73.33, 75.75, 78.14]
# Model_A = [53.00, 57.83, 56.20, 68.57]
# Model_B = [71.00, 74.67, 76.20, 78.71]
# Model_C = [72.44,73.03, 74.96, 75.69]
# Model_D = [86.13, 84.54, 83.81, 87.79]


# X축 값과 레이블 설정
x = [0, 1, 2, 3]
labels = ["10 Labels", "30 Labels", "50 Labels", "70 Labels"]

plt.figure(dpi=100, figsize=(11, 6))

# plt.plot(x, finetuning_1D, marker='o', markersize=6, linestyle='-', label="1D Finetuning", linewidth=3, color='#ff7f0e')
# plt.plot(x, freeze_1D, marker='o', markersize=6, linestyle='-', label="1D Freeze", linewidth=3, color='#2ca02c')
# plt.plot(x, finetuning_2D, marker='o', markersize=6, linestyle='--', label="2D Finetuning", linewidth=3, color='#1f77b4')
# plt.plot(x, freeze_2D, marker='o', markersize=6, linestyle='--', label="2D Freeze", linewidth=3, color='#d62728')
#
plt.plot(x, Model_A, marker='o', markersize=6, linestyle='--', label="Model A", linewidth=3, color='darkgreen')
plt.plot(x, Model_B, marker='o', markersize=6, linestyle='-', label="Model B", linewidth=3, color='darkgreen')
plt.plot(x, Model_C, marker='o', markersize=6, linestyle='--', label="Model C", linewidth=3, color='darkorange')
plt.plot(x, Model_D, marker='o', markersize=6, linestyle='-', label="Model D", linewidth=3, color='darkorange')
# plt.plot(x, Model_A, marker='o', markersize=6, linestyle='--', label="1D Model C", linewidth=3, color='#1f77b4')
# plt.plot(x, Model_B, marker='o', markersize=6, linestyle='-', label="1D Model D", linewidth=3, color='#1f77b4')
# plt.plot(x, Model_C, marker='o', markersize=6, linestyle='--', label="2D Model C", linewidth=3, color='firebrick')
# plt.plot(x, Model_D, marker='o', markersize=6, linestyle='-', label="2D Model D", linewidth=3, color='firebrick')

plt.xticks(x, labels, fontsize=15)
plt.yticks([10.00,20.00,30.00,40.00,50.00, 60.00, 70.00, 80.00, 90.00], fontsize=15)
plt.gca().set_yticklabels([f"{y:.2f}%" for y in plt.gca().get_yticks()])
plt.ylim(10, 90)
# plt.rc("legend", fontsize="medium", fontsize=80)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4, prop=font_prop, fontsize="large")

output_file_path = "C:/Users/user/Desktop/연구실/시각화/GRAPH/comparison_plot_4.png"
plt.savefig(output_file_path, format='png', bbox_inches='tight')
plt.close()
print(f"{output_file_path}에 그래프 저장 완료")


# #################
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# # 한국어 폰트 설정
# font_path = "C:\\Windows\\Fonts\\malgun.ttf"
# font_prop = fm.FontProperties(fname=font_path, size=15)
#
# data_size = [10, 30, 50, 70]
# finetuning_1D = [53.00, 57.83, 56.20, 68.57]
# freeze_1D = [71.00, 74.67, 76.20, 78.71]
# finetuning_2D = [70.36, 68.10, 70.14, 70.56]
# freeze_2D = [82.50, 80.36, 79.79, 84.42]
#
# # X축 값과 레이블 설정
# x = [0, 1, 2, 3]
# labels = ["10 Labels", "30 Labels", "50 Labels", "70 Labels"]
#
# plt.figure(dpi=100, figsize=(13, 6))
#
# colors = ['red', 'blue', 'green', 'purple']
#
# plt.plot(x, finetuning_1D, marker='o', markersize=6, linestyle='-', label="1D Finetuning", linewidth=3, color='#ff7f0e')
# plt.plot(x, freeze_1D, marker='o', markersize=6, linestyle='-', label="1D Freeze", linewidth=3, color='#2ca02c')
# plt.plot(x, finetuning_2D, marker='o', markersize=6, linestyle='--', label="2D Finetuning", linewidth=3, color='#1f77b4')
# plt.plot(x, freeze_2D, marker='o', markersize=6, linestyle='--', label="2D Freeze", linewidth=3, color='#d62728')
#
# plt.xticks(x, labels, fontsize=15)
# plt.yticks([50, 60, 70, 80, 90], fontsize=15)
#
# plt.gca().set_yticklabels([f"{int(y)}%" for y in plt.gca().get_yticks()])
#
# plt.ylim(50, 90)
#
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4, prop=font_prop, fontsize=30)
#
# output_file_path = "C:\\Users\\gyuhee\\Desktop\\비교 PLOT\\comparison_plot.png"
# plt.savefig(output_file_path, format='png', bbox_inches='tight')
# plt.close()
# print(f"{output_file_path}에 그래프 저장 완료")
#
#
# # finetuning_1D = [57.50, 70.00, 73.00, 75.35]
# # freeze_1D = [62.00, 66.33, 73.20, 75.43]
# # finetuning_2D = [69.17, 75.83, 73.75, 75.49]
# # freeze_2D = [71.25, 70.14, 73.33, 74.64]

# # Model_A = [57.50, 70.00, 73.00, 75.35]
# # Model_B = [62.00, 66.33, 73.20, 75.43]
# # Model_C = [69.17, 75.83, 73.75, 75.49]
# # Model_D = [71.25, 70.14, 73.33, 74.64]
#Model_A = [53.00, 57.83, 56.20, 68.57]
#Model_B = [71.00, 74.67, 76.20, 78.71]
#Model_C = [70.36, 68.10, 70.14, 70.56]
#Model_D = [82.50, 80.36, 79.79, 84.42]