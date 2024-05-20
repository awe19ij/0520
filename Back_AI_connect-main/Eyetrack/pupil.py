import numpy as np
import cv2


class Pupil(object):
#눈동자를 감지하고 눈동자의 위치를 추정

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        #눈 프레임에서 홍채를 격리
        #eye_frame (numpy.ndarray): 눈만 포함하는 프레임
        #threshold (int): 눈 프레임을 이진화하기 위해 사용되는 임계값
        
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        # 새 창에 홍채를 표시--------------------
        #cv2.imshow("Iris Frame", new_frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #---------------------------------------

        return new_frame #홍채만 있는 프레임

    def detect_iris(self, eye_frame):
        #홍채를 감지하고 홍채의 위치를 추정, 무게 중심을 계산하여 눈의 위치를 추정
        #eye_frame (numpy.ndarray): 눈만 포함하는 프레임
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])

        except (IndexError, ZeroDivisionError):
            pass
