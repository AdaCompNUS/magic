from data_store import DataClient
import cv2
import numpy as np

GOAL = (1058, 428)
GOAL_RADIUS = 80
NOISY_REGIONS = [
    [(590, 40), (700, 558)],
    [(790, 40), (900, 558)],
]

if __name__ == '__main__':
    dc = DataClient()

    while True:
        p = dc['perception']
        if p is None:
            exit()

        frame = dc['frame']

        if p['ball'] is not None:
            cv2.drawMarker(frame,
                    tuple(np.int0(p['ball'][0])),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    50, 2)

        if p['bot'] is not None:
            cv2.line(frame,
                    tuple(np.int0(p['bot'][0])),
                    tuple(np.int0(p['bot'][0] + 50 * p['bot'][1])),
                    (255, 255, 0),
                    2)

        for region in NOISY_REGIONS:
            frame = cv2.rectangle(frame, region[0], region[1], (0, 255, 0), 1)

        frame = cv2.circle(frame, GOAL, GOAL_RADIUS, (0, 255, 0), 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('{}, {} -> {}'.format(x, y, frame[y, x]))

        cv2.imshow('frame', frame)
        cv2.setMouseCallback("frame", mouse_callback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
