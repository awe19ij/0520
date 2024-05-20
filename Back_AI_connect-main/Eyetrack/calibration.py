from __future__ import division
import cv2
from .pupil import Pupil


class Calibration(object):
    #사람과 웹캠에 가장 적합한 이진화 임계값을 찾아 동공 감지 알고리즘을 보정

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        #보정이 완료되었는지 여부를 반환
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        #주어진 눈에 대한 임계값을 반환
        #side: 왼쪽 눈인지 (0) 오른쪽 눈인지 (1)
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        #눈의 표면에서 홍채가 차지하는 공간의 백분율을 반환
        #frame (numpy.ndarray): 이진화된 홍채 프레임
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        #주어진 눈의 프레임에 대한 최적의 임계값을 계산
        #eye_frame (numpy.ndarray): 분석할 눈의 프레임
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        #주어진 이미지를 고려하여 보정을 개선
        #eye_frame (numpy.ndarray): 눈의 프레임
        #side: 왼쪽 눈 (0) 또는 오른쪽 눈 (1)을 나타냅니다.
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
