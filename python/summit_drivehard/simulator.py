import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--port-base', type=int, default=23000)
parser.add_argument('--mode', type=str, default='handcrafted')
parser.add_argument('--model-path', type=str)
parser.add_argument('--macro-length', type=int, default=1,
                    help='Macro-action length')
parser.add_argument('--debug', help='Print debug info', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--visualize', help='Visualize runs', action='store_true')
parser.set_defaults(debug=False)
args = parser.parse_args()

from controller import Controller
from environment import Environment
import Pyro4
import argparse
import glob
import math
import multiprocessing
import numpy as np
import os
import signal
import subprocess
import sys
import time
import traceback
import zmq
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from models import MAGICGenNet_DriveHard, MAGICCriticNet_DriveHard, RLActorNet_DriveHard
from utils import Statistics, PARTICLE_SIZES, CONTEXT_SIZES
import random

sys.path.append(glob.glob(os.path.expanduser('~/summit/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
import carla

DEBUG = args.debug
VISUALIZE = args.visualize

MAP = 'meskel_square'
TASK = 'DriveHard'
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] if TASK in CONTEXT_SIZES else None
NUM_EXO_AGENTS = 15
EXO_AGENT_PREF_SPEED = 4.0
EGO_AGENT_MAX_SPEED = 6.0
AGENT_MIN = [-0.53535, -0.98701]
AGENT_MAX = [3.2668, 0.98701]
MAX_STEER = np.deg2rad(70)

ZMQ_ADDRESS = 'tcp://127.0.0.1'
SUMMIT_PATH = os.path.expanduser('~/summit/CarlaUE4.sh')
SUMMIT_SCRIPTS_PATH = os.path.expanduser('~/summit/PythonAPI/examples')
SUMMIT_DATA_PATH = os.path.expanduser('~/summit/Data')

EGO_AGENT_PATH_RESOLUTION = 0.1
EGO_PATH_LENGTH = 1500
CONTEXT_PATH_INTERVAL = 10
AGENT_TARGET_LOOKAHEAD = 5.0
PROGRESS_REWARD_WEIGHT = 1.0
COLLISION_REWARD = -100
GAMMA = 0.98
SEARCH_DEPTH = 40
LOW_SPEED_THRESHOLD = EGO_AGENT_MAX_SPEED / 2.0;
LOW_SPEED_PENALTY = -1.0 * COLLISION_REWARD * (1 - GAMMA) / (1 - GAMMA**SEARCH_DEPTH);
MAX_STEPS = 150
DELTA = 0.2
MACRO_LENGTH = args.macro_length

def debug_print(text, *wargs):
    if DEBUG:
        print(text, *wargs, file=sys.stderr)

def get_position(actor):
    pos3d = actor.get_location()
    return carla.Vector2D(pos3d.x, pos3d.y)

def get_heading(actor):
    heading = actor.get_transform().get_forward_vector()
    return carla.Vector2D(heading.x, heading.y)

def get_speed(actor):
    v = actor.get_velocity()
    return np.linalg.norm([v.x, v.y])

def get_steer(actor):
    ctrl = actor.get_control()
    return ctrl.steer * MAX_STEER

def get_bounding_box_corners(actor, inflate=0):
    forward = get_heading(actor)
    sideward = forward.rotate(np.pi / 2)
    pos = get_position(actor)

    return [
          pos + (AGENT_MIN[0] - inflate) * forward + (AGENT_MIN[1] - inflate) * sideward,
          pos + (AGENT_MIN[0] - inflate) * forward + (AGENT_MAX[1] + inflate) * sideward,
          pos + (AGENT_MAX[0] + inflate) * forward + (AGENT_MAX[1] + inflate) * sideward,
          pos + (AGENT_MAX[0] + inflate) * forward + (AGENT_MIN[1] - inflate) * sideward
    ]



def environment_process(port):
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    if args.mode == 'magic':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, True, True).float().to(device)
        gen_model.load_state_dict(torch.load(args.model_path))
        gen_model_lambda = lambda context, state: gen_model.mode(
                torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    elif args.mode == 'sac':
        if MACRO_LENGTH != 1:
            debug_print('WARNING: Overriding macro-length with 1 in mode=pomcpow')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gen_model = RLActorNet_DriveHard(0, 111).float().to(device)
        gen_model.load_state_dict(torch.load(args.model_path))

    if args.mode == 'pomcpow':
        if MACRO_LENGTH != 1:
            debug_print('WARNING: Overriding macro-length with 1 in mode=pomcpow')
        env = Environment(1)
    else:
        env = Environment(MACRO_LENGTH)

    try:

        update_and_plan_update_failed = False

        while True:
            request = socket.recv_pyobj()
            instruction = request[0]
            instruction_data = request[1:]
            debug_print('[PY][{}]'.format(instruction))

            if instruction == 'INITIALIZE_BELIEF':
                env.initialize_belief(instruction_data[0], instruction_data[1])
                socket.send_pyobj(0)
            elif instruction == 'UPDATE_BELIEF':
                if update_and_plan_update_failed:
                    socket.send_pyobj('TERMINAL')
                else:
                    env.update_belief(instruction_data[0], instruction_data[1])
                    socket.send_pyobj(0) # Return non-terminal status of current belief.
                    #env.debug_belief(instruction_data[1])
            elif instruction == 'POP_UPDATE_BELIEF':
                result = env.pop_update_belief()
                debug_print('[PY][{}] -> {}'.format(instruction, result))
                socket.send_pyobj(result)
            elif instruction == 'PLAN':
                socket.send_pyobj(0)
                if args.mode == 'handcrafted':
                    env.forward_plan_handcrafted(instruction_data[0])
                elif args.mode == 'magic':
                    env.forward_plan_magic(instruction_data[0], gen_model_lambda)
                elif args.mode == 'pomcpow':
                    env.forward_plan_pomcpow(instruction_data[0])
                else:
                    raise Exception('Unknown mode!')
            elif instruction == 'UPDATE_BELIEF_AND_FORWARD_PLAN':
                socket.send_pyobj(0)
                env.update_belief(instruction_data[0], instruction_data[1])
                result = env.pop_update_belief()
                debug_print('[PY][{}] -> {}'.format(instruction, result))
                if result == 'TERMINAL':
                    update_and_plan_update_failed = True
                else:
                    if args.mode == 'handcrafted':
                        env.forward_plan_handcrafted(instruction_data[2])
                    elif args.mode == 'magic':
                        env.forward_plan_magic(instruction_data[2], gen_model_lambda)
                    elif args.mode == 'pomcpow':
                        env.forward_plan_pomcpow(instruction_data[2])
                    else:
                        raise Exception('Unknown mode!')
            elif instruction == 'POP_FORWARD_PLAN_RESULT':
                if update_and_plan_update_failed:
                    debug_print('[PY][{}] -> update_and_plan_update_failed = True'.format(instruction))
                    socket.send_pyobj('TERMINAL')
                else:
                    result = env.pop_forward_plan_result()
                    debug_print('[PY][{}] -> {}'.format(instruction, result))
                    socket.send_pyobj(result)
            elif instruction == 'UPDATE_BELIEF_AND_SAC':
                env.update_belief(instruction_data[0], instruction_data[1])
                result = env.pop_update_belief()
                debug_print('[PY][{}] -> {}'.format(instruction, result))
                if result == 'TERMINAL':
                    socket.send_pyobj('TERMINAL')
                else:
                    state = env.sample_belief()
                    state = tuple(torch.tensor(s, dtype=torch.float32).cuda().unsqueeze(0) for s in [[], state]) # (context, state)
                    with torch.no_grad():
                        action = gen_model.sample(state)[0][0].cpu().numpy()
                        action = [EGO_AGENT_MAX_SPEED * (action[0] + 1.0) / 2.0, action[1] * np.pi / 3]
                    socket.send_pyobj(action)
            elif instruction == 'DEBUG_FORWARD_BELIEF':
                socket.send_pyobj(0)
            elif instruction == 'TERMINATE':
                socket.send_pyobj(0)
                break

        env.terminate()

    except:
        pass
    finally:
        socket.close()
        zmq_context.term()


class Observation:

    def __init__(self):
        self.ego_agent_position = None
        self.ego_agent_heading = None
        self.ego_agent_speed = None
        self.ego_agent_steer = None
        self.exo_agent_positions = [None for _ in range(NUM_EXO_AGENTS)]
        self.exo_agent_headings = [None for _ in range(NUM_EXO_AGENTS)]
        self.exo_agent_speeds = [None for _ in range(NUM_EXO_AGENTS)]
        self.exo_agent_steers = [None for _ in range(NUM_EXO_AGENTS)]

    def serialize(self):
        data = []
        data.extend([
            self.ego_agent_position.x, self.ego_agent_position.y,
            self.ego_agent_heading.x, self.ego_agent_heading.y,
            self.ego_agent_speed, self.ego_agent_steer])

        for i in range(NUM_EXO_AGENTS):
            data.extend([
                self.exo_agent_positions[i].x, self.exo_agent_positions[i].y,
                self.exo_agent_headings[i].x, self.exo_agent_headings[i].y,
                self.exo_agent_speeds[i], self.exo_agent_steers[i]])

        return data



class CarlaSimulator:

    def __init__(self, port, pyro_port):
        self.port = port
        self.pyro_port = pyro_port
        self.carla_client = None
        self.carla_world = None
        self.crowd_service = None

        self.simulator_process = None
        self.spawn_mesh_process = None
        self.spawn_imagery_process = None
        self.gamma_crowd_process = None
        self.controller = None
        self.start_time = None

        self.steps = 0
        self.distance = 0

        with open('{}/{}.sim_bounds'.format(SUMMIT_DATA_PATH, MAP), 'r') as f:
            self.bounds_min = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
            self.bounds_max = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
            self.bounds_occupancy = carla.OccupancyMap(self.bounds_min, self.bounds_max)
        self.sumo_network = carla.SumoNetwork.load('{}/{}.net.xml'.format(SUMMIT_DATA_PATH, MAP))
        self.sumo_network_segments = self.sumo_network.create_segment_map()
        self.sumo_network_spawn_segments = self.sumo_network_segments.intersection(carla.OccupancyMap(self.bounds_min, self.bounds_max))
        self.sumo_network_occupancy = self.sumo_network.create_occupancy_map()
        self.ego_actor = None
        self.exo_actor_indexes = None # Actor id to index mapping.

        self.junction_occupancy = carla.OccupancyMap([
            carla.Vector2D(471.03, 378.36), carla.Vector2D(483.46, 410.76), carla.Vector2D(459.30, 443.16), carla.Vector2D(427.59, 443.02),
            carla.Vector2D(424.94, 441.77), carla.Vector2D(402.31, 418.72), carla.Vector2D(426.48, 376.40), carla.Vector2D(467.54, 374.73)
        ]).intersection(self.sumo_network_occupancy)

        self.entry_occupancies = [
            carla.OccupancyMap([
                carla.Vector2D(470.33, 378.38),
                carla.Vector2D(474.53, 390.83),
                carla.Vector2D(482.48, 390.08),
                carla.Vector2D(483.38, 377.78)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(460.58, 441.70),
                carla.Vector2D(446.47, 442.00),
                carla.Vector2D(446.17, 454.16),
                carla.Vector2D(459.68, 454.61)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(425.51, 441.43),
                carla.Vector2D(413.80, 429.64),
                carla.Vector2D(402.52, 440.31),
                carla.Vector2D(413.54, 452.62)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(426.29, 376.91),
                carla.Vector2D(451.70, 376.50),
                carla.Vector2D(453.78, 361.39),
                carla.Vector2D(428.80, 357.66)]).intersection(self.sumo_network_occupancy)
        ]

        self.exit_occupancies = [
            carla.OccupancyMap([
                carla.Vector2D(478.26, 398.58),
                carla.Vector2D(487.70, 397.91),
                carla.Vector2D(556.91, 402.38),
                carla.Vector2D(555.77, 415.20),
                carla.Vector2D(483.01, 411.01)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(440.81, 442.21),
                carla.Vector2D(432.38, 552.29),
                carla.Vector2D(419.03, 551.95),
                carla.Vector2D(427.31, 442.10)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(402.50, 417.68),
                carla.Vector2D(414.85, 430.84),
                carla.Vector2D(340.70, 500.13),
                carla.Vector2D(327.72, 485.45)]).intersection(self.sumo_network_occupancy),
            carla.OccupancyMap([
                carla.Vector2D(467.79, 375.95),
                carla.Vector2D(450.02, 376.41),
                carla.Vector2D(468.49, 248.63),
                carla.Vector2D(486.84, 253.13)]).intersection(self.sumo_network_occupancy)
        ]

    def terminate(self):
        if self.controller is not None:
            self.controller.terminate()

        for p in [self.simulator_process, self.spawn_mesh_process, self.spawn_imagery_process, self.gamma_crowd_process]:
            if p is not None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)


    def launch(self):
        debug_print('Launching simulator process...')
        self.simulator_process = subprocess.Popen(
            [
                SUMMIT_PATH,
                '-carla-port={}'.format(self.port),
                '-quality-level={}'.format('Epic' if VISUALIZE else 'Low'),
                '-opengl'
            ],
            env=dict(os.environ, SDL_VIDEODRIVER='' if VISUALIZE else 'offscreen'),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        debug_print('    Delaying for simulator to start up...')
        time.sleep(5)
        debug_print('    Creating client...')
        self.carla_client = carla.Client('127.0.0.1', self.port)

    def reset_world(self):
        debug_print('Resetting world...')
        self.carla_client.reload_world()
        self.carla_world = self.carla_client.get_world()

        # Spawn meshes.
        debug_print('    Spawning meshes...')
        self.spawn_mesh_process = subprocess.Popen(
            [
                'python3',
                SUMMIT_SCRIPTS_PATH + '/spawn_meshes.py',
                '--dataset={}'.format(MAP),
                '--port={}'.format(self.port)
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        self.spawn_mesh_process.wait()
        self.spawn_mesh_process = None

        self.spawn_imagery_process = subprocess.Popen(
            [
                'python3',
                SUMMIT_SCRIPTS_PATH + '/spawn_imagery.py',
                '--dataset={}'.format(MAP),
                '--port={}'.format(self.port)
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        self.spawn_imagery_process.wait()
        self.spawn_imagery_process = None

        '''
        # For calibrating transform
        calib_pos = [(1, 1.5), (1, -1.5), (-1, 1.5), (-1, -1.5)]
        calib_pos = [(50 * x, 50 * y) for (x, y) in calib_pos]
        calib_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for (pos, color) in zip(calib_pos, calib_color):
            bounds_mid = (self.bounds_min + self.bounds_max) / 2
            loc = carla.Location()
            print(bounds_mid)
            loc.x = bounds_mid.x + pos[0]
            loc.y = bounds_mid.y + pos[1]
            loc.z = 1
            print((loc.x, loc.y), color)
            self.carla_world.debug.draw_point(loc, size=0.05, color=carla.Color(*color), life_time=-1)
        '''

        entry_occupancy = self.entry_occupancies[random.randint(0, len(self.entry_occupancies) - 1)]
        entry_segments = self.sumo_network_segments.intersection(entry_occupancy)
        ego_start = entry_segments.rand_point()
        ego_rp = self.sumo_network.get_nearest_route_point(ego_start)

        '''
        ego_start = carla.Vector2D(409.0, 437.0)
        ego_rp = self.sumo_network.get_nearest_route_point(ego_start)
        entry_occupancy = [o for o in self.entry_occupancies if o.contains(ego_start)][0]
        entry_segments = self.sumo_network_segments.intersection(entry_occupancy)
        '''

        self.ego_path = [ego_rp]
        path_rp = ego_rp
        exit_occupancy = None
        for i in range(1, EGO_PATH_LENGTH):
            next_rps = self.sumo_network.get_next_route_points(path_rp, EGO_AGENT_PATH_RESOLUTION)
            path_rp = next_rps[random.randint(0, len(next_rps) - 1)]
            self.ego_path.append(path_rp)
            if exit_occupancy is None:
                for occupancy in self.exit_occupancies:
                    if occupancy.contains(self.sumo_network.get_route_point_position(path_rp)):
                        exit_occupancy = occupancy
                        break

        self.ego_sumo_network_occupancy = self.junction_occupancy.union(entry_occupancy).union(exit_occupancy)

        # Spawn ego-agent.
        debug_print('    Spawning ego-agent...')
        self.ego_rp_pos = self.sumo_network.get_route_point_position(ego_rp)
        ego_next_pos = self.sumo_network.get_route_point_position(self.sumo_network.get_next_route_points(ego_rp, 1.0)[0])
        ego_trans = carla.Transform()
        ego_trans.location.x = self.ego_rp_pos.x
        ego_trans.location.y = self.ego_rp_pos.y
        ego_trans.location.z = 0.2
        ego_trans.rotation.yaw = np.rad2deg(math.atan2(ego_next_pos.y - self.ego_rp_pos.y, ego_next_pos.x - self.ego_rp_pos.x))
        ego_blueprint = self.carla_world.get_blueprint_library().filter('vehicle.mini.cooperst')[0]
        self.ego_actor = self.carla_world.try_spawn_actor(ego_blueprint, ego_trans)
        self.ego_actor.set_collision_enabled(False)
        self.ego_progress_index = 0

        actor_physics_control = self.ego_actor.get_physics_control()
        actor_physics_control.gear_switch_time = 0.0
        self.ego_actor.apply_physics_control(actor_physics_control)

        for i in range(EGO_PATH_LENGTH - 1):
            loc1 = carla.Location()
            pos1 = self.sumo_network.get_route_point_position(self.ego_path[i])
            (loc1.x, loc1.y, loc1.z) = (pos1.x, pos1.y, 0.1)
            loc2 = carla.Location()
            pos2 = self.sumo_network.get_route_point_position(self.ego_path[i + 1])
            (loc2.x, loc2.y, loc2.z) = (pos2.x, pos2.y, 0.1)
            self.carla_world.debug.draw_line(
                    loc1, loc2,
                    thickness=0.2, color=carla.Color(255, 153, 51),
                    persistent_lines=True)

        # Launch controller.
        debug_print('        Launching controller...')
        self.controller = Controller(self.port, self.ego_actor.id)

        # Launch gamma crowd.
        debug_print('    Launching GAMMA process...')
        self.gamma_crowd_process = subprocess.Popen(
            [
                'python3',
                SUMMIT_SCRIPTS_PATH + '/gamma_crowd.py',
                '--dataset={}'.format(MAP),
                '--num-car={}'.format(NUM_EXO_AGENTS),
                '--num-bike=0',
                '--num-pedestrian=0',
                '--no-respawn',
                '--seed={}'.format(int(time.time())),
                '--speed-car={}'.format(EXO_AGENT_PREF_SPEED),
                '--port={}'.format(self.port),
                '--pyroport={}'.format(self.pyro_port),
                '--aim-center'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid)
        self.crowd_service = Pyro4.Proxy('PYRO:crowdservice.warehouse@localhost:{}'.format(self.pyro_port))
        debug_print('        Delaying for crowd service to launch...')
        time.sleep(3)
        debug_print('        Waiting for spawn target to be reached...')
        while not self.crowd_service.spawn_target_reached:
            time.sleep(0.2)

        # Assign actor id to indexes.
        time.sleep(1) # To allow agent list to refresh.
        self.exo_actor_indexes = dict()
        index = 0
        for actor in self.carla_world.get_actors():
            if isinstance(actor, carla.Vehicle) and actor.id != self.ego_actor.id:
                self.exo_actor_indexes[actor.id] = index
                index += 1
        if len(self.exo_actor_indexes) != NUM_EXO_AGENTS:
            raise Exception('ERROR: Incorrect number of exo agents!')

    def observe(self):
        obs = Observation()
        obs.ego_agent_position = get_position(self.ego_actor)
        obs.ego_agent_heading = get_heading(self.ego_actor)
        obs.ego_agent_speed = get_speed(self.ego_actor)
        obs.ego_agent_steer = get_steer(self.ego_actor)

        # Get fixed-index list.
        exo_actors = [None for _ in range(NUM_EXO_AGENTS)]
        for actor in self.carla_world.get_actors():
            if isinstance(actor, carla.Vehicle) and actor.id != self.ego_actor.id:
                exo_actors[self.exo_actor_indexes[actor.id]] = actor

        for (i, exo_actor) in enumerate(exo_actors):
            if exo_actor is not None:
                obs.exo_agent_positions[i] = get_position(exo_actor)
                obs.exo_agent_headings[i] = get_heading(exo_actor)
                obs.exo_agent_speeds[i] = get_speed(exo_actor)
                obs.exo_agent_steers[i] = get_steer(exo_actor)

        return obs

    def context(self):
        c = []
        for p in self.ego_path:
            pos = self.sumo_network.get_route_point_position(p)
            c.extend([pos.x, pos.y])
        return c

    def step(self, action):
        if self.steps == 0:
            self.start_time = time.time()

        self.controller.set_target(action)
        #time.sleep(max(MIN_STEP_TIME, (self.steps + 1) * DELTA - (time.time() - self.start_time)))
        time.sleep(DELTA)
        obs = self.observe()

        # Step ego agent progress.
        rp_steps = 0
        result_index = self.ego_progress_index
        result_rp_pos = self.sumo_network.get_route_point_position(self.ego_path[self.ego_progress_index])
        while result_index < len(self.ego_path) - 2:
            next_rp = self.ego_path[result_index + 1]
            next_rp_pos = self.sumo_network.get_route_point_position(next_rp)
            if (get_position(self.ego_actor) - result_rp_pos).length() < (get_position(self.ego_actor) - next_rp_pos).length():
                break
            result_index += 1
            result_rp_pos = next_rp_pos
            rp_steps += 1
        self.ego_progress_index = result_index
        self.distance += rp_steps * EGO_AGENT_PATH_RESOLUTION
        self.steps += 1

        # Calculate rewards.
        progress = rp_steps * EGO_AGENT_PATH_RESOLUTION
        speed = get_speed(self.ego_actor)
        steer = abs(get_steer(self.ego_actor))
        steer_factor = (speed / EGO_AGENT_MAX_SPEED) * (steer**2)
        speed_penalty = 1 if speed < LOW_SPEED_PENALTY else 0
        low_speed_penalty = LOW_SPEED_PENALTY if speed < LOW_SPEED_THRESHOLD else 0
        reward = PROGRESS_REWARD_WEIGHT * progress - low_speed_penalty

        # Calculate terminal.
        has_collision = False
        terminal = False
        failure = False

        ego_bb = carla.OccupancyMap(get_bounding_box_corners(self.ego_actor, -0.1))
        for exo_actor in self.carla_world.get_actors():
            if isinstance(exo_actor, carla.Vehicle) and exo_actor.id != self.ego_actor.id:
                exo_bb = carla.OccupancyMap(get_bounding_box_corners(exo_actor, -0.1))
                if not has_collision and ego_bb.intersects(exo_bb):
                    has_collision = True
        if has_collision:
            reward = COLLISION_REWARD
            terminal = True
            failure = True
            debug_print('COLLISION')

        if not terminal and not ego_bb.difference(self.sumo_network_occupancy).is_empty:
            reward = COLLISION_REWARD
            terminal = True
            failure = True
            debug_print('OOB')

        if not terminal and self.steps >= MAX_STEPS:
            terminal = True
            debug_print('STEPS')

        return (reward, obs, terminal, failure, progress, speed, steer, steer_factor, speed_penalty)



if __name__ == '__main__':


    debug_print('Launching environment process...')
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    zmq_port = socket.bind_to_random_port(ZMQ_ADDRESS)
    env_p = multiprocessing.Process(
            target=environment_process,
            args=(zmq_port,),
            daemon=False)
    env_p.start()

    sim = CarlaSimulator(args.port_base, args.port_base + 10)
    sim.launch()
    sim.reset_world()
    debug_print('Launched')

    debug_print('Initializing environment...')
    obs = sim.observe()
    socket.send_pyobj(('INITIALIZE_BELIEF', obs.serialize(), sim.context()))
    socket.recv_pyobj()
    action_queue = [(0.0, 0.0) for _ in range(1 if args.mode in ['pomcpow', 'sac'] else MACRO_LENGTH)]

    '''
    if not os.path.exists('sync'):
        os.mkdir('sync')
    timings_file = open('sync/timings.log', 'w')
    ego_data_file = open('sync/ego_data.log', 'w')
    '''

    # To debug belief tracking.
    start = time.time()
    total_reward = 0
    total_progress = 0
    episode_failure = False
    speed_m1 = 0
    speed_m2 = 0
    steer_m1 = 0
    steer_m2 = 0
    steer_factor_m1 = 0
    steer_factor_m2 = 0
    steer_max = None
    speed_penalty_m1 = 0
    speed_penalty_m2 = 0
    max_depth = 0
    initial_step_done = False
    requires_pop_update_belief = False

    start_signal_loc = carla.Location()
    start_signal_loc.x = ((sim.bounds_min + sim.bounds_max) / 2).x - 50
    start_signal_loc.y = ((sim.bounds_min + sim.bounds_max) / 2).y - 90
    start_signal_loc.z = 1
    sim.carla_world.debug.draw_point(start_signal_loc, size=0.5, color=carla.Color(255, 0, 0), life_time=-1)

    try:

        while True:
            debug_print('@@@@@ STEP START @@@@@') # Step start

            if not initial_step_done:
                if args.mode not in ['sac']:
                    socket.send_pyobj(('PLAN', action_queue))
                    socket.recv_pyobj()
                initial_step_done = True

            # Get action and pop.
            action = action_queue[0]
            action_queue = action_queue[1:]

            # Print debug info.
            debug_print('S = {}, T = {:.2f} / {:.2f}, A = {}'.format(
                sim.steps - 1, time.time() - start,
                sim.steps * DELTA, action))

            '''
            # Output synchronization info.
            timings_file.write('{}\n'.format(time.time()))
            ego_pos = get_position(sim.ego_actor)
            ego_heading = get_heading(sim.ego_actor)
            ego_speed = get_speed(sim.ego_actor)
            ego_data_file.write('{}, {}, {}, {}, {}\n'.format(ego_pos.x, ego_pos.y, ego_heading.x, ego_heading.y, ego_speed))
            '''
            (reward, obs, terminal, failure, progress, speed, steer, steer_factor, speed_penalty) = sim.step(action)

            # Step sim.
            debug_print('    R = {:.2f}, P = {:.2f}, Speed = {:.2f}, Steer = {:.2f}, SF = {:.2f}, SP = {} TERM = {}'.format(
                reward, progress, speed, steer, steer_factor, speed_penalty, terminal))
            total_reward += reward
            total_progress += progress
            episode_failure = episode_failure or failure
            speed_m1 += speed
            speed_m2 += speed**2
            steer_m1 += steer
            steer_m2 += steer**2
            steer_factor_m1 += steer_factor
            steer_factor_m2 += steer_factor**2
            speed_penalty_m1 += speed_penalty
            speed_penalty_m2 += speed_penalty**2
            steer_max = steer if steer_max is None else max(steer, steer_max)

            if terminal:
                break

            if args.mode not in ['sac']:
                if len(action_queue) == 0:

                    # Pop previous result and extend action queue.
                    socket.send_pyobj(('POP_FORWARD_PLAN_RESULT',))
                    result = socket.recv_pyobj()
                    if isinstance(result, str) and result == 'TERMINAL':
                        debug_print('BREAK 1')
                        break
                    action_queue.extend(result[0])
                    max_depth = max(max_depth, result[1])

                    # Complete previous belief update.
                    if requires_pop_update_belief:
                        socket.send_pyobj(('POP_UPDATE_BELIEF',))
                        result = socket.recv_pyobj()
                        if isinstance(result, str) and result == 'TERMINAL':
                            debug_print('BREAK 2')
                            break
                    requires_pop_update_belief = False

                    # Update belief and start forward plan.
                    # Assumes update sub-step can complete before next delta.
                    socket.send_pyobj(('UPDATE_BELIEF_AND_FORWARD_PLAN', action, obs.serialize(), action_queue))
                    socket.recv_pyobj()

                else:
                    # Complete previous belief update.
                    if requires_pop_update_belief:
                        socket.send_pyobj(('POP_UPDATE_BELIEF',))
                        result = socket.recv_pyobj()
                        if isinstance(result, str) and result == 'TERMINAL':
                            debug_print('BREAK 3')
                            break
                    requires_pop_update_belief = False

                    # Trigger belief update.
                    socket.send_pyobj(('UPDATE_BELIEF', action, obs.serialize()))
                    # Rare condition: When running UPDATE_BELIEF_AND_FORWARD_PLAN before this, the update belief sub-step can lead to
                    # a terminal belief. However, we aren't notified prior to UPDATE_BELIEF: this is to catch that condition.
                    result = socket.recv_pyobj()
                    if isinstance(result, str) and result == 'TERMINAL':
                        break
                    requires_pop_update_belief = True
            else:
                socket.send_pyobj(('UPDATE_BELIEF_AND_SAC', action, obs.serialize()))
                result = socket.recv_pyobj()
                if isinstance(result, str) and result == 'TERMINAL':
                    break
                action_queue.append(result)

        socket.send_pyobj(('TERMINATE',))
        socket.recv_pyobj()

        print(total_reward, total_progress,
                speed_m1, speed_m2,
                steer_m1, steer_m2,
                steer_factor_m1, steer_factor_m2,
                speed_penalty_m1, speed_penalty_m2,
                max_depth,
                steer_max, sim.steps, 1 if episode_failure else 0)

    except:
        pass
    finally:
        sim.terminate()
        socket.close()
        zmq_context.term()
        exit()
