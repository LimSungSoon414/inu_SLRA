#############################g히믜아희망
import os
import numpy as np
from fastdtw import fastdtw
from shutil import copy2

def calculate_dtw_distance(target_prototype, source_data):
    # Ensure the data is 1-dimensional
    target_prototype = target_prototype.flatten()
    source_data = source_data.flatten()
    distance, path = fastdtw(target_prototype, source_data)
    return distance

# 타겟 데이터 위치와 클래스
target_path = r"C:\Users\user\Desktop\연구실\NPY\6208_numpy"
target_classes = ['N', 'SB', 'SO', 'SI']
num_samples_per_class = 15

# 타겟 도메인에서 15개의 데이터 선택
target_data = []
target_labels = []

print("타겟 데이터 로딩 시작.")
for target_class in target_classes:
    target_class_path = os.path.join(target_path, target_class)
    target_class_files = os.listdir(target_class_path)[:num_samples_per_class]

    for file in target_class_files:
        file_path = os.path.join(target_class_path, file)
        data = np.load(file_path)
        target_data.append(data)
        target_labels.append(target_classes.index(target_class))
    print(f"{target_class} 클래스 타겟 데이터 로딩 완료.")


target_data_list = np.array(target_data)
target_labels_list = np.array(target_labels)

# 각 클래스별 대표 프로토타입 저장
target_prototypes = []
print("타겟 데이터 프로토타입 계산 시작.")
for i in range(len(target_classes)):
    target_prototype = np.mean(target_data_list[target_labels_list == i], axis=0)
    target_prototypes.append(target_prototype)
    print(f"{target_classes[i]} 클래스 프로토타입: {target_prototype}")

# 소스 데이터
source_path = r"C:\Users\user\Desktop\연구실\NPY\6207_numpy"
source_classes = ['N', 'SB', 'SO', 'SI']
num_samples_per_class_source = 3000

# 소스 도메인 각 클래스 별로 1000개씩 데이터셋을 리스트 저장
source_data = []
source_labels = []

print("소스 데이터 로딩 시작.")
# # 수정
selected_files = []  # 각 클래스별로 선택된 파일 목록을 저장할 리스트 추가
for source_class in source_classes:
    source_class_path = os.path.join(source_path, source_class)
    source_class_files = os.listdir(source_class_path)[:num_samples_per_class_source]

    for file in source_class_files:
        file_path = os.path.join(source_class_path, file)
        data = np.load(file_path)
        source_data.append(data)
        source_labels.append(source_classes.index(source_class))
    print(f"{source_class} 클래스 소스 데이터 로딩 완료.")
    selected_files.append(source_class_files)  # 해당 클래스의 목록 저장
# # 수정된 부분 끝

source_data_list = np.array(source_data)
source_labels_list = np.array(source_labels)

# 유사성 계산
similarities = []
print("유사성 계산 시작.")
for i in range(len(source_classes)):
    source_data_class = source_data_list[source_labels_list == i]
    target_prototype = target_prototypes[i]

    #
    print(f"클래스 {source_classes[i]}의 유사성 계산 중... (총 {len(source_data_class)}개 데이터)")

    similarity = np.array([calculate_dtw_distance(target_prototype, source_data) for source_data in source_data_class])
    similarities.append(similarity)
    print(f"{source_classes[i]} 클래스 유사성 계산 완료.")

# 유사성이 높은 상위
selected_files = [[] for _ in range(len(source_classes))]
print("유사성 높은 데이터 선택 시작.")
for i in range(len(similarities)):
    similarity = similarities[i]
    top_700_indices = np.argsort(similarity)[:1000]
    selected_files_class = source_class_files[top_700_indices]  # 수정된 부분
    selected_files[i] = selected_files_class
    print(f"{source_classes[i]} 클래스 데이터 선택 완료.")

#
destination_folder = r"C:\Users\gyuhee\Desktop\연구실\similarity\CWRU_integrate"
print("선택된 데이터 복사 시작.")
for i in range(len(selected_files)):
    selected_files_class = selected_files[i]
    destination_class_folder = os.path.join(destination_folder, source_classes[i])

    if not os.path.exists(destination_class_folder):
        os.makedirs(destination_class_folder)

    for file in selected_files_class:
        source_file_path = os.path.join(source_path, source_classes[i], file)
        destination_file_path = os.path.join(destination_class_folder, file)

        #
        if os.path.exists(source_file_path):
            copy2(source_file_path, destination_file_path)
            print(f"{source_file_path} 복사 완료.")
        else:
            print(f"경고: {source_file_path} 파일이 존재하지 않아 복사하지 않았습니다.")
print("모든 과정 완료.")
# print(source_class_files)