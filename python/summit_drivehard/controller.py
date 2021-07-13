import glob
import math
import multiprocessing
import numpy as np
import os
import sys
import time
import zmq

sys.path.append(glob.glob(os.path.expanduser('~/summit/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
import carla

ZMQ_ADDRESS = 'tcp://127.0.0.1'
CONTROL_MAX_RATE = 20.0
STEER_MAX = 70 * np.pi / 180

# Ziegler-Nichols: some overshoot.
SPEED_KU = 0.8
SPEED_TU = 20.0 / 14
SPEED_KP = 0.33333 * SPEED_KU
SPEED_KI = 0.66666 * SPEED_KU / SPEED_TU
SPEED_KD = 0.11111 * SPEED_KU * SPEED_TU

try:
    sys.path.append(glob.glob(os.path.expanduser('~/summit/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass
import carla

def get_velocity(actor):
    v = actor.get_velocity()
    return carla.Vector2D(v.x, v.y)

def get_signed_angle_diff(vector1, vector2):
    theta = math.atan2(vector1.y, vector1.x) - math.atan2(vector2.y, vector2.x)
    theta = np.rad2deg(theta)
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

def get_heading(actor):
    heading = actor.get_transform().get_forward_vector()
    return carla.Vector2D(heading.x, heading.y)

def controller_loop(carla_port, zmq_port, ego_actor_id):
    carla_client = carla.Client('127.0.0.1', carla_port)
    carla_world = carla_client.get_world()
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1) # Keep only last message in queue.
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, zmq_port))
    ego_actor = carla_world.get_actor(ego_actor_id)

    target_speed = 0
    target_steer = 0
    last_update_time = None
    speed_integral = 0
    last_speed_error = None

    while True:
        heading = get_heading(ego_actor)
        line_start = ego_actor.get_location()
        line_start.z = 4.0
        line_end = carla.Location()
        line_end.x = line_start.x + 4 * heading.x
        line_end.y = line_start.y + 4 * heading.y
        line_end.z = 4.0
        '''
        carla_world.debug.draw_line(
                line_start, line_end, thickness=0.3, color=carla.Color(255, 0, 0),
                life_time=0.1, persistent_lines=False)
        '''

        start = time.time()
        if last_update_time is None:
            dt = 0
        else:
            dt = start - last_update_time

        try:
            request = socket.recv_pyobj(flags=zmq.NOBLOCK)
            instruction = request[0]
            instruction_data = request[1:]

            if instruction == 'SET_TARGET':
                target_speed = float(instruction_data[0][0])
                target_steer = float(instruction_data[0][1])
            elif instruction == 'TERMINATE':
                break
        except zmq.Again as e:
            pass

        # Get current values.
        vel = get_velocity(ego_actor)
        speed = vel.length()

        # Calculate speed error.
        speed_error = np.clip(target_speed, speed - 1.0, speed + 1.0) - speed

        # Add to integral. Clip to stablize integral term.
        speed_integral += np.clip(speed_error, -0.3 / SPEED_KP, 0.3 / SPEED_KP) * dt

        # Calculate output.
        speed_control = SPEED_KP * speed_error + SPEED_KI * speed_integral
        if last_update_time is not None:
            speed_control += SPEED_KD * (speed_error - last_speed_error) / dt

        # Update last errors.
        last_speed_error = speed_error

        # Apply control.
        control = ego_actor.get_control()
        if speed_control >= 0:
            control.throttle = speed_control
            control.brake = 0.0
            control.hand_brake = False
        else:
            control.throttle = 0.0
            control.brake = -speed_control
            control.hand_brake = False
        control.steer = np.clip(target_steer / STEER_MAX, -1.0, 1.0)
        if control.gear == 0:
            control.manual_gear_shift = True
            control.gear = 1
        else:
            control.manual_gear_shift = False
        ego_actor.apply_control(control)

        # Update last update time.
        last_update_time = start

        time.sleep(max(0, 1 / CONTROL_MAX_RATE - (time.time() - start)))


class Controller():

    def __init__(self, carla_port, ego_actor_id):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        zmq_port = self.socket.bind_to_random_port(ZMQ_ADDRESS)

        self.controller_process = multiprocessing.Process(
                target=controller_loop,
                args=(carla_port, zmq_port, ego_actor_id),
                daemon=False)
        self.controller_process.start()
        time.sleep(1) # Delay for process to start up.

    def set_target(self, action):
        self.socket.send_pyobj(('SET_TARGET', action))

    def terminate(self):
        self.socket.send_pyobj(('TERMINATE',))
