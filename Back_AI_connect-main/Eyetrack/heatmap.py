import cv2
import numpy as np
import pandas as pd

# 이미지와 CSV 파일 경로 설정
image_path = "C:/KJE/IME_graduation_AI/AI_JE/0509/image.png"
csv_path = "C:/KJE/IME_graduation_AI/AI_JE/0509/gaze_sections.csv"

# 중심에서 바깥쪽으로 그라데이션 적용하는 함수
def apply_gradient(center, inner_radius, outer_radius, color, image):
    overlay = image.copy()
    cv2.circle(overlay, center, inner_radius, (255, 255, 255), -1)  # 안쪽 원 (흰색)
    cv2.circle(overlay, center, outer_radius, color, -1)  # 바깥쪽 원 (그라데이션)
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

# 히트맵 그리기 함수
def draw_heatmap(image, section_counts):
    if image is not None:
        # 이미지의 가로 및 세로 크기
        height, width, _ = image.shape

        # 각 영역의 중심 좌표
        section_centers = {
            "A": (int(width / 6), int(height / 4)),
            "B": (int(width / 2), int(height / 4)),
            "C": (int(5 * width / 6), int(height / 4)),
            "D": (int(width / 6), int(3 * height / 4)),
            "E": (int(width / 2), int(3 * height / 4)),
            "F": (int(5 * width / 6), int(3 * height / 4))
        }

        # 각 영역에 대한 히트맵 그리기
        for section, count in section_counts.items():
            if section in section_centers:
                center = section_centers[section]
                # 카운트 값을 퍼센트로 변환하여 원의 반지름 계산
                percent = int(100 * (count / max(section_counts.values())))  # 전체 대비 해당 영역 응시 비율 계산
                
                # 여러 겹의 원 그리기
                for i in range(1, 5):  # 4개의 원 그리기
                    inner_radius = 30 * i
                    outer_radius = 30 * i + percent  # 바깥쪽 원의 반지름
                    
                    # 색상 조절
                    if i == 1:
                        color = (0, 0, 255)  # 빨강
                    elif i == 2:
                        color = (0, 200, 200)  # 노랑
                    elif i == 3:
                        color = (0, 200, 0)  # 초록
                    elif i == 4:
                        color = (200, 0, 0)  # 파랑
                    
                    # 그라데이션 적용
                    apply_gradient(center, inner_radius, outer_radius, color, image)

# 이미지 불러오기
image = cv2.imread(image_path)

# CSV 파일에서 데이터 불러오기
section_data = pd.read_csv(csv_path)
section_counts = dict(zip(section_data["Section"], section_data["Count"]))

# 히트맵 그리기
draw_heatmap(image, section_counts)

# 결과 이미지 출력 및 저장
cv2.imshow("Heatmap", image)
cv2.imwrite("C:/KJE/IME_graduation_AI/AI_JE/0509/heatmap_output.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
