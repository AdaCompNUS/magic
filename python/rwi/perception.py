import os
os.environ['OMP_NUM_THREADS'] = '1' # Multithreading doesn't really help for our pipeline.
import cv2
cv2.setNumThreads(1) # Multithreading doesn't really help for our pipeline.

import numpy as np
import time
from data_store import DataClient
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

CV2_CAPTURE_PATH = '/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_8C8FB2EF-video-index0'
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters_create()
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
BOT_ARUCO_ID = 16

BOARD_WIDTH = 1280
BOARD_HEIGHT = 587
BALL_RADIUS = 16.0
NOISY_REGIONS = [
    [(590, 40), (700, 558)],
    [(790, 40), (900, 558)],
]

def rotated(v, angle):
    return np.array([
        np.cos(angle) * v[0] - np.sin(angle) * v[1],
        np.sin(angle) * v[0] + np.cos(angle) * v[1]], dtype=np.float32)

def perspectiveTransform(s, m):
    s = np.matmul(m, np.array([s[0], s[1], 1]))
    s = np.array([s[0] / s[2], s[1] / s[2]])
    return s

def in_noisy_regions(p):
    for r in NOISY_REGIONS:
        if p[0] >= r[0][0] - BALL_RADIUS and \
                p[0] <= r[1][0] + BALL_RADIUS and \
                p[1] >= r[0][1] - BALL_RADIUS and \
                p[1] <= r[1][1] + BALL_RADIUS:
            return True
    return False

def get_ball(frame):
    for r in NOISY_REGIONS:
        ball_frame = cv2.rectangle(frame, tuple(np.int0(r[0])), tuple(np.int0(r[1])),
                    (0, 0, 0), -1)
    ball_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_frame = cv2.inRange(ball_frame, (10, 60, 80), (40, 200, 200))
    ball_frame = cv2.dilate(ball_frame, (3, 3))
    ball_frame = cv2.erode(ball_frame, (3, 3))
    (ball_contours, ball_hierarchy) = cv2.findContours(ball_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_eccentricity = None
    best_ball = None
    for (i, contour) in enumerate(ball_contours):
        m = cv2.moments(contour, False)
        if abs(m['m00']) < 0.001:
            continue
        mp02 = m['mu20'] / m['m00']
        mp20 = m['mu02'] / m['m00']
        mp11 = m['mu11'] / m['m00']
        a = (mp20 + mp02) / 2
        b = np.sqrt(4 * (mp11**2) + (mp20 - mp02)**2) / 2
        l1 = a + b
        l2 = a - b
        eccentricity = np.sqrt(1 - l2 / l1)

        ball = cv2.minEnclosingCircle(contour)
        if ball[1] > 7 and eccentricity < 0.65 and not in_noisy_regions(ball[0]):
            if best_eccentricity is None or eccentricity < best_eccentricity:
                best_ball = ball
                best_eccentricity = eccentricity

    return best_ball

def get_bot(frame):
    (corners, ids, rejectedImgPoints) = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    if ids is not None:
        for (i, idx) in enumerate(ids):
            if idx == BOT_ARUCO_ID:
                corners = corners[i].squeeze()
                center = np.mean(corners, axis=0)
                forward = np.mean(corners[[0, 1]], axis=0) - center
                forward /= np.linalg.norm(forward)
                return (center, forward)
    return None

def bot_filter_f(x, dt):
    v = x[4]
    omega = x[5]

    if abs(omega) < 0.01:
        return np.array([
            x[0] + dt * v * x[2],
            x[1] + dt * v * x[3],
            x[2],
            x[3],
            x[4],
            x[5]], dtype=np.float32)
    else:
        r = v / omega
        pos = x[0:2]
        heading = x[2:4]

        rotation_center = pos + r * rotated(heading, np.pi / 2)
        pos = rotated(pos - rotation_center, omega * dt) + rotation_center
        heading = rotated(heading, omega * dt)

        return np.array([pos[0], pos[1], heading[0], heading[1], x[4], x[5]])

def bot_filter_h(x):
    return x[0:4]

class Perception(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(CV2_CAPTURE_PATH, apiPreference=cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with open('calibration/board_transform.npy', 'rb') as f:
            self.board_transform = np.load(f)
        with open('calibration/bot_transform.npy', 'rb') as f:
            self.bot_transform = np.load(f)

        self.bot_filter = UnscentedKalmanFilter(
                dim_x=6, dim_z=4,
                dt=None, fx=bot_filter_f, hx=bot_filter_h,
                points=MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1))
        self.bot_filter.P = np.diag([5**2, 5**2, 0.1**2, 0.1**2, 1**2, 0.1**2])
        self.bot_filter.Q = np.array([
            [(5)**2, 0, 0, 0, 0, 0],
            [0, (5)**2, 0, 0, 0, 0],
            [0, 0, (0.1)**2, 0, 0, 0],
            [0, 0, 0, (0.1)**2, 0, 0],
            [0, 0, 0, 0, (200)**2, 0],
            [0, 0, 0, 0, 0, (1)**2]], dtype=np.float32)
        self.bot_filter.R = np.diag([5**2, 5**2, 0.1**2, 0.1**2])
        self.bot_filter_last_update = None

        self.dc = DataClient()

    def start(self):
        recent_frame_times = []
        last_time = None
        previous_ball_position = None
        while True:
            dc_update = dict()

            (ret, frame) = self.cap.read()
            frame_time = time.time()
            transformed_frame = cv2.warpPerspective(np.copy(frame), self.board_transform, (BOARD_WIDTH, BOARD_HEIGHT))
            dc_update['frame_time'] = frame_time

            ball = get_ball(cv2.GaussianBlur(transformed_frame, (3, 3), 0))
            dc_update['ball'] = ball

            #frame_count += 1
            #print(frame_count/ (time.time() -start))

            bot_position = None
            bot_heading = None
            bot = get_bot(frame)
            if bot is not None: # Has current -> initialize if no previous, else predict and update.
                (bot_position, bot_heading) = bot
                bot_position = perspectiveTransform(bot[0], self.bot_transform)
                bot_heading = perspectiveTransform(bot[0] + 100 * bot[1], self.bot_transform)
                bot_heading -= bot_position
                bot_heading /= np.linalg.norm(bot_heading)


                if self.bot_filter_last_update is None:
                    self.bot_filter.x = np.array([
                        bot_position[0], bot_position[1],
                        bot_heading[0], bot_heading[1],
                        0, 0], dtype=np.float32)
                else:
                    self.bot_filter.predict(frame_time - self.bot_filter_last_update)
                    self.bot_filter.update(np.array([bot_position[0], bot_position[1], bot_heading[0], bot_heading[1]]))
                self.bot_filter_last_update = frame_time
                dc_update['bot'] = (self.bot_filter.x[0:2], self.bot_filter.x[2:4], self.bot_filter.x[4:6])
            elif self.bot_filter_last_update is not None: # Has previous, missing current -> predict.
                self.bot_filter.predict(frame_time - self.bot_filter_last_update)
                self.bot_filter_last_update = frame_time
                dc_update['bot'] = (self.bot_filter.x[0:2], self.bot_filter.x[2:4], self.bot_filter.x[4:6])
            else: # No current and no previous.
                dc_update['bot'] = None

            self.dc['perception'] = dc_update
            self.dc['frame'] = transformed_frame

            last_time = frame_time

if __name__ == '__main__':
    Perception().start()
