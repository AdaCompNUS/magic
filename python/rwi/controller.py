import numpy as np
import time
from data_store import DataClient
import do_mpc
from simple_pid import PID
from rc_client import RCClient

DELTA = 2.0
ACTION_LOOKAHAEAD_DIST = 50
ACTION_TRACK_WIDTH = 60

def rotated(v, angle):
    return np.array([
        np.cos(angle) * v[0] - np.sin(angle) * v[1],
        np.sin(angle) * v[0] + np.cos(angle) * v[1]], dtype=np.float32)

def angle_to(v_from, v_to):
    return np.arctan2(
            v_from[0] * v_to[1] - v_from[1] * v_to[0],
            v_from[0] * v_to[0] + v_from[1] * v_to[1])

class Controller(object):

    def __init__(self):
        self.dc = DataClient()
        self.rc = RCClient()

        self.mpc_turn_model = do_mpc.model.Model('continuous')
        theta = self.mpc_turn_model.set_variable('_x', 'theta', shape=(1, 1))
        omega = self.mpc_turn_model.set_variable('_u', 'omega', shape=(1, 1))
        self.mpc_turn_model.set_rhs('theta', omega)
        self.mpc_turn_model.setup()
        self.mpc_turn = do_mpc.controller.MPC(self.mpc_turn_model)
        setup_mpc_turn = {
            'n_horizon': 10,
            't_step': 0.2,
            'n_robust': 0,
            'store_full_solution': False
        }
        self.mpc_turn.set_param(**setup_mpc_turn)
        self.mpc_turn.set_objective(mterm=theta**2, lterm=theta**2 + 0.06 * omega**2)
        self.mpc_turn.set_rterm(omega=6e-2)
        self.mpc_turn.bounds['lower', '_x', 'theta'] = -np.pi
        self.mpc_turn.bounds['upper', '_x', 'theta'] = np.pi
        self.mpc_turn.bounds['lower', '_u', 'omega'] = -6 * np.pi
        self.mpc_turn.bounds['upper', '_u', 'omega'] = 6 * np.pi
        self.mpc_turn.set_param(nlpsol_opts={'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
        self.mpc_turn.setup()
        self.mpc_turn.set_initial_guess()

        self.bot_intended_ray_start = None
        self.bot_intended_heading = None

        self.previous_action = None

        while self.dc['perception'] is None:
            time.sleep(0.01)

    def turn(self, orientation, callback=None):
        if self.previous_action != 'TURN':
            self.rc.send(0, 0)

        self.bot_intended_ray_start = None
        self.bot_intended_heading = np.array([np.cos(orientation), np.sin(orientation)])

        start = time.time()
        while True:
            if time.time() - start >= DELTA:
                break
            p = self.dc['perception']
            bot_position = p['bot'][0]
            bot_heading = p['bot'][1]

            if self.bot_intended_ray_start is None:
                self.bot_intended_ray_start = bot_position

            theta = angle_to(bot_heading, self.bot_intended_heading)
            control = self.mpc_turn.make_step(np.array([theta]))
            control = (0.5 * ACTION_TRACK_WIDTH * control / 9).item() # PWM -> SPEED : multiply by 9
            if abs(control) > 0.5 and abs(control) < 3:
                control = np.sign(control) * 3
            self.rc.send(-control, control)
            if callback is not None:
                callback()

        self.previous_action = 'TURN'

    def move(self, distance, callback=None):
        start = time.time()
        while True:
            p = self.dc['perception']
            bot_position = p['bot'][0]
            bot_heading = p['bot'][1]

            if time.time() - start >= DELTA:
                break

            shadow = np.dot(bot_position - self.bot_intended_ray_start, self.bot_intended_heading)
            if abs(shadow - distance)  < 1:
                self.rc.send(0, 0)
            else:
                p_base = (distance - shadow) * 0.12
                p_base = np.sign(p_base) * min(abs(p_base), 15)

                if p_base >= 0:
                    lookahead_point = self.bot_intended_ray_start + (shadow + ACTION_LOOKAHAEAD_DIST) * self.bot_intended_heading
                else:
                    lookahead_point = self.bot_intended_ray_start + (shadow - ACTION_LOOKAHAEAD_DIST) * self.bot_intended_heading
                offset = lookahead_point - bot_position
                d_hor = np.dot(rotated(bot_heading, np.pi / 2), offset)
                d_ver = np.dot(bot_heading, offset)
                d_radius = (d_hor**2 + d_ver**2) / (2 * abs(d_hor))
                d_dir = np.sign(d_hor)
                p_outer = (p_base * (1 + 0.5 * ACTION_TRACK_WIDTH / d_radius))
                p_inner = (p_base * (1 - 0.5 * ACTION_TRACK_WIDTH / d_radius))

                if d_dir >= 0:
                    (p_left, p_right) = (p_outer, p_inner)
                else:
                    (p_left, p_right) = (p_inner, p_outer)

                self.rc.send(p_left, p_right)

            if callback is not None:
                callback()

        self.previous_action = 'MOVE'

    def stop(self):
        self.rc.send(0, 0)
        time.sleep(DELTA)
        self.previous_action = 'STOP'
