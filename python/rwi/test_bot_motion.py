from controller import Controller
from data_store import DataClient
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import time
from scipy.optimize import least_squares
import matplotlib

if __name__ == '__main__':

    dc = DataClient()
    controller = Controller()

    ACTIONS = [0, np.pi]

    for _ in range(30):
        controller.turn(ACTIONS[0])

    action_index = 0
    results = []
    while True:

        action = ACTIONS[action_index]

        timings = []
        positions = []
        timings2 = []
        positions2 = []
        start = time.time()
        last_frame_time = None

        def record(output_timings, output_positions):
            global start
            global last_frame_time
            p = dc['perception']
            if p is not None and p['bot'] is not None:
                if last_frame_time is not None and p['frame_time'] == last_frame_time:
                    return
                bot = p['bot']
                output_timings.append(p['frame_time'] - start)
                output_positions.append(bot[0])
                last_frame_time = p['frame_time']

        for _ in range(30):
            controller.move(lambda: record(timings, positions))

        for _ in range(30):
            controller.turn(action + 1, lambda: record(timings2, positions2))

        print('Step 1:')
        print(timings)
        print(positions)
        print('Step 2:')
        print(timings2)
        print(positions2)

        ray_start = np.array(positions[0])
        ray_dir = np.array(positions[-1]) - np.array(positions[0])
        ray_dir /= np.linalg.norm(ray_dir)
        positions = np.dot(np.array(positions) - ray_start, ray_dir)
        positions2 = np.dot(np.array(positions2) - ray_start, ray_dir)

        fig, ax = plt.subplots()
        plt.plot(timings, positions, label='Motion')
        plt.plot(timings2, positions2, label='Brake')
        plt.legend()
        plt.show()
        continue

        def fun(x, t):
            A = x[0]
            B = x[1]
            s_0 = x[2]
            v_0 = x[3]

            return s_0 + (A / B) * t + (v_0 - A / B) / B * (1 - np.exp(-B * t))

        def fun_residue(x, t, y):
            return fun(x, t) - y

        fit = least_squares(
                fun_residue,
                np.array([10, 0.1, positions[0], (positions[5] - positions[0]) / timings[5]], dtype=np.float64),
                args=(timings, positions),
                jac='3-point', loss='linear',
                bounds=([-2000.0, -20.0, 0, -500], [2000.0, 20.0, 20, 500]),
                max_nfev=10000)

        A = abs(fit.x[0])
        B = fit.x[1]
        results.append([A, B])


        print('Iteration: {}'.format(len(results)))
        print('Result: A = {}, B = {}'.format(A, B))
        print('Average: A = {}, B = {}'.format(*np.mean(results, axis=0)))

        action_index = (action_index + 1) % len(ACTIONS)
