from __future__ import division
import os
import cv2
import dlib
import numpy as np
from .eye import Eye
from .calibration import Calibration
from math import hypot

loc1 = 0
loc2 = 0

class GazeTracking(object):
    #사용자의 시선을 추적
    #눈의 위치와 동공과 같은 유용한 정보를 제공, 눈이 열려 있는지 닫혀 있는지를 제공

    def __init__(self):
        self.frame = None
        self.test_frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector는 얼굴을 감지하는 데 사용
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor는 주어진 얼굴의 얼굴 특징점을 얻는 데 사용
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        #동공이 위치했는지 확인
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        #얼굴을 감지하고 Eye 객체를 초기화
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        #프레임을 새로 고치고 분석
        #frame (numpy.ndarray): 분석할 프레임
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        #왼쪽 동공의 좌표를 반환
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return x, y

    def pupil_right_coords(self):
        #오른쪽 동공의 좌표를 반환
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return x, y

    def horizontal_ratio(self):
        #시선의 수평 방향을 나타내는 0.0에서 1.0 사이의 수를 반환
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        #시선의 수직 방향을 나타내는 0.0에서 1.0 사이의 수를 반환
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        else:
            return True

    def is_top_right(self, avg_right_hor_gaze, avg_top_ver_gaze):
        #사용자가 우상단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            # return self.horizontal_ratio() <= 0.65 and self.vertical_ratio() <= 0.9
            return self.horizontal_ratio() <= avg_right_hor_gaze and self.vertical_ratio() <= avg_top_ver_gaze
        else:
            return True

    def is_top_left(self, avg_left_hor_gaze, avg_top_ver_gaze):
        #사용자가 좌상단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            # return self.horizontal_ratio() >= 0.78 and self.vertical_ratio() <= 0.9
            return self.horizontal_ratio() >= avg_left_hor_gaze and self.vertical_ratio() <= avg_top_ver_gaze
        else:
            return True

    def is_top_center(self, avg_top_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
        #사용자가 중앙상단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            return self.is_top_right(avg_right_hor_gaze, avg_top_ver_gaze) is not True and self.is_top_left(avg_left_hor_gaze, avg_top_ver_gaze) is not True and self.vertical_ratio() <= avg_top_ver_gaze
        else:
            return True

    def is_bottom_center(self, avg_bottom_ver_gaze, avg_right_hor_gaze, avg_left_hor_gaze):
        #사용자가 중앙하단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            return self.is_bottom_right(avg_right_hor_gaze, avg_bottom_ver_gaze) is not True and self.is_bottom_left(avg_left_hor_gaze, avg_bottom_ver_gaze) is not True and self.vertical_ratio() > avg_bottom_ver_gaze
        else:
            return True

    def is_bottom_right(self, avg_right_hor_gaze, avg_bottom_ver_gaze):
        #사용자가 우하단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            # return self.horizontal_ratio() <= 0.65 and self.vertical_ratio() > 0.9
            return self.horizontal_ratio() <= avg_right_hor_gaze and self.vertical_ratio() > avg_bottom_ver_gaze
        else:
            return True

    def is_bottom_left(self, avg_left_hor_gaze, avg_bottom_ver_gaze):
        #사용자가 좌하단을 보고 있는 경우 true를 반환
        if self.pupils_located:
            # return self.horizontal_ratio() >= 0.78 and self.vertical_ratio() > 0.9
            return self.horizontal_ratio() >= avg_left_hor_gaze and self.vertical_ratio() > avg_bottom_ver_gaze
        else:
            return True

    def is_blinking(self):
       #사용자가 눈을 감았을 때 true를 반환
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 5.7

    def annotated_frame(self):
        #동공이 강조된 메인 프레임을 반환
        frame = self.frame.copy()
        global loc1
        global loc2

        if self.pupils_located:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for face in self._face_detector(gray):
                loc1 = (face.left(), face.top())
                loc2 = (face.right(), face.bottom())

                landmarks = self._predictor(gray, face)
                landmarks_points = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))

                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                points = np.array(landmarks_points, np.int32)
                convexhull = cv2.convexHull(points)
                cv2.polylines(frame, [convexhull], True, (255, 0, 0), 1)

                rect = cv2.boundingRect(convexhull)
                subdiv = cv2.Subdiv2D(rect)
                subdiv.insert(landmarks_points)
                triangles = subdiv.getTriangleList()
                triangles = np.array(triangles, dtype=np.int32)

                for t in triangles:
                    pt1 = (t[0], t[1])
                    pt2 = (t[2], t[3])
                    pt3 = (t[4], t[5])

                    cv2.line(frame, pt1, pt2, (0, 0, 255), 1)
                    cv2.line(frame, pt2, pt3, (0, 0, 255), 1)
                    cv2.line(frame, pt1, pt3, (0, 0, 255), 1)

            left_eye_Lpoint = (landmarks.part(36).x, landmarks.part(36).y)
            left_eye_Rpoint = (landmarks.part(39).x, landmarks.part(39).y)
            right_eye_Lpoint = (landmarks.part(42).x, landmarks.part(42).y)
            right_eye_Rpoint = (landmarks.part(45).x, landmarks.part(45).y)
            hor_line_len1 = hypot((left_eye_Lpoint[0] - left_eye_Rpoint[0]), (left_eye_Lpoint[1] - left_eye_Rpoint[1]))
            hor_line_len2 = hypot((right_eye_Lpoint[0] - right_eye_Rpoint[0]), (right_eye_Lpoint[1] - right_eye_Rpoint[1]))
            color = (0, 255, 0)
            color1 = (255, 0, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            cv2.circle(frame, (int((left_eye_Lpoint[0] + left_eye_Rpoint[0]) / 2), int((left_eye_Lpoint[1] + left_eye_Rpoint[1]) / 2)), int(hor_line_len1/2), color1, 1)
            cv2.circle(frame, (int((right_eye_Lpoint[0] + right_eye_Rpoint[0]) / 2), int((right_eye_Lpoint[1] + right_eye_Rpoint[1]) / 2)), int(hor_line_len2/2), color1, 1)

            # print(self.vertical_ratio())

        return frame, loc1, loc2
