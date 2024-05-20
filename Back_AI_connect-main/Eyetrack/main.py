import cv2
import numpy as np
import csv
import time
import threading
import keyboard
from gaze_tracking import GazeTracking

end_sig = False
esc = False

webcam = cv2.VideoCapture(0)
R_top = 0
L_top = 0
C_top = 0

R_bottom = 0
L_bottom = 0
C_bottom = 0

avg_left_hor_gaze = 0
avg_right_hor_gaze = 0
avg_top_ver_gaze = 0
avg_bottom_ver_gaze = 0

total_left_hor_gaze = 0
total_right_hor_gaze = 0
total_top_ver_gaze = 0
total_bottom_ver_gaze = 0

sectionA = 0
sectionB = 0
sectionC = 0
sectionD = 0
sectionE = 0
sectionF = 0
section = "None"

# 각 섹션의 횟수를 기록할 딕셔너리
sections = {
    "A": sectionA,
    "B": sectionB,
    "C": sectionC,
    "D": sectionD,
    "E": sectionE,
    "F": sectionF
}

gaze_time = 0
test_count = 1
flag = 0
gaze = GazeTracking()

def Section(where):
    global sectionA, sectionB, sectionC, sectionD, sectionE, sectionF
    if where == "A":
        sections["A"] += 1
        return sections["A"]
    elif where == "B":
        sections["B"] += 1
        return sections["B"]
    elif where == "C":
        sections["C"] += 1
        return sections["C"]
    elif where == "D":
        sections["D"] += 1
        return sections["D"]
    elif where == "E":
        sections["E"] += 1
        return sections["E"]
    elif where == "F":
        sections["F"] += 1
        return sections["F"]

def Thread_run():
    global section
    print(section, ":", Section(section))
    thread = threading.Timer(0.1, Thread_run)#0.1초 단위 기록
    thread.daemon = True
    thread.start()
    return thread

thread = Thread_run()

while True:
    if keyboard.is_pressed('q'):
        break

    _, frame = webcam.read()
    gaze.refresh(frame)
    frame, loc1, loc2 = gaze.annotated_frame()

    text = ""

    if test_count < 50:
        cv2.circle(frame, (25, 25), 25, (0, 0, 255), -1)
        if gaze.horizontal_ratio() != None and gaze.vertical_ratio() != None:
            total_left_hor_gaze += gaze.horizontal_ratio()
            total_top_ver_gaze += gaze.vertical_ratio()
            test_count += 1
    elif 50 <= test_count < 100:
        cv2.circle(frame, (610, 25), 25, (0, 0, 255), -1)
        if gaze.horizontal_ratio() != None and gaze.vertical_ratio() != None:
            total_right_hor_gaze += gaze.horizontal_ratio()
            total_top_ver_gaze += gaze.vertical_ratio()
            test_count += 1
    elif 100 <= test_count < 150:
        cv2.circle(frame, (25, 450), 25, (0, 0, 255), -1)
        if gaze.horizontal_ratio() != None and gaze.vertical_ratio() != None:
            total_left_hor_gaze += gaze.horizontal_ratio()
            total_bottom_ver_gaze += gaze.vertical_ratio()
            test_count += 1
    elif 150 <= test_count < 200:
        cv2.circle(frame, (610, 450), 25, (0, 0, 255), -1)
        if gaze.horizontal_ratio() != None and gaze.vertical_ratio() != None:
            total_right_hor_gaze += gaze.horizontal_ratio()
            total_bottom_ver_gaze += gaze.vertical_ratio()
            test_count += 1

    else:
        if flag == 0:
            avg_left_hor_gaze = total_left_hor_gaze / 100
            avg_right_hor_gaze = total_right_hor_gaze / 100
            avg_top_ver_gaze = total_top_ver_gaze / 100
            avg_bottom_ver_gaze = total_bottom_ver_gaze / 100
            flag = 1

        if gaze.is_blinking():
            text = "Blinking"

        if gaze.is_top_left(avg_left_hor_gaze, avg_top_ver_gaze):
            cv2.putText(frame, "Top Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking top left"
            section = "A"
        elif gaze.is_top_center(avg_top_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
            cv2.putText(frame, "Top Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking top center"
            section = "B"
        elif gaze.is_top_right(avg_right_hor_gaze, avg_top_ver_gaze):
            cv2.putText(frame, "Top Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking top right"
            section = "C"
        elif gaze.is_bottom_left(avg_left_hor_gaze, avg_top_ver_gaze):
            cv2.putText(frame, "Bottom Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking bottom left"
            section = "D"
        elif gaze.is_bottom_center(avg_top_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
            cv2.putText(frame, "Bottom Center", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking bottom center"
            section = "E"
        elif gaze.is_bottom_right(avg_right_hor_gaze, avg_top_ver_gaze):
            cv2.putText(frame, "Bottom Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = "Looking bottom right"
            section = "F"
        gaze_time = int(time.time())
        save_loc1 = loc1
        save_loc2 = loc2

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:#esc눌러서 저장하고 종료
        # CSV 파일로 데이터 저장
        csv_filename = "C:/KJE/IME_graduation_AI/AI_JE/0509/gaze_sections.csv"

        # CSV 파일 헤더
        csv_header = ["Section", "Count"]

        # CSV 파일 쓰기 모드로 열기
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # 헤더 쓰기
            writer.writerow(csv_header)
            
            # 각 섹션의 횟수를 CSV 파일에 기록
            for section_name, count in sections.items():
                writer.writerow([section_name, count])

        print("Data saved to", csv_filename)
        break

total_gaze = 0
#print("Top Gaze ratio : ", round(R_top / total_gaze, 2), round(L_top / total_gaze, 2), round(C_top / total_gaze, 2))
#print("Bottom Gaze ratio: ", round(R_bottom / total_gaze, 2), round(L_bottom / total_gaze, 2), round(C_bottom / total_gaze, 2))
cv2.destroyAllWindows()