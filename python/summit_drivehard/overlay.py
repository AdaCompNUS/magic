import cv2
import numpy as np
from base64 import b64encode, b64decode
import struct

EPISODE_SIGNAL_POS = (222, 982)

DELTA = 0.2
EGO_AGENT_ACCEL = 10000.0
AGENT_WHEEL_BASE = 2.5106
MACRO_LENGTH = 3
EGO_AGENT_SPEED_MAX = 6.0
EGO_AGENT_STEER_MAX = 15.0 * np.pi / 180.0

'''
COLORS = [
        (75, 25, 230),
        (180, 30, 145),
        (75, 180, 60),

        (25, 225, 255),
        (230, 50, 240),
        (49, 130, 245),

        (240, 240, 70),
        (216, 99, 67)]
'''

'''
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (255, 255, 255)
]
'''

COLORS = [(39, 121, 233)]


def rotate(x, d):
    return np.array([x[0] * np.cos(d) - x[1] * np.sin(d), x[0] * np.sin(d) + x[1] * np.cos(d)], dtype=np.float32)

def actuate(agent_position, agent_heading, agent_speed, control_speed, control_steer):

    accel = EGO_AGENT_ACCEL

    if agent_speed > control_speed:
        accel = -accel

    accel_time = min(DELTA, (control_speed - agent_speed) / accel)
    travelled_distance = agent_speed * accel_time + 0.5 * accel * accel_time * accel_time + speed * (DELTA - accel_time)

    if abs(control_steer) < 0.0001:
        next_agent_position = agent_position + agent_heading * travelled_distance
        next_agent_heading = agent_heading
    else:
        radius = AGENT_WHEEL_BASE / np.arctan(steer)
        center = agent_position + radius * rotate(agent_heading, np.pi / 2)
        angle = travelled_distance / radius
        next_agent_position = center + rotate(agent_position - center, angle)
        next_agent_heading = rotate(agent_heading, angle)
        next_agent_heading = next_agent_heading / np.linalg.norm(next_agent_heading)

    next_agent_speed = agent_speed + accel * accel_time

    return (next_agent_position, next_agent_heading, next_agent_speed)

'''
DriveHard::Agent DriveHard::Actuate(const Agent& agent, float speed, float accel, float steer) {
  if (agent.speed > speed) {
    accel = -accel;
  }

  Agent next_agent = agent;
  float accel_time = std::min(DELTA, (speed - agent.speed) / accel);
  float travelled_distance = // v0t0 + 0.5at0^2 + v1t1
      agent.speed * accel_time + 0.5f * accel * accel_time * accel_time
      + speed * (DELTA - accel_time);

  if (std::abs(steer) < 0.0001f) {
    next_agent.position = agent.position + agent.heading * travelled_distance;
    next_agent.heading = agent.heading;
  } else {
    float radius = AGENT_WHEEL_BASE  / atanf(steer);
    vector_t center = agent.position + radius * agent.heading.rotated(PI / 2);
    float angle = travelled_distance / radius;

    next_agent.position = center + (agent.position - center).rotated(angle);
    next_agent.heading = agent.heading.rotated(angle).normalized();
  }

  next_agent.speed = agent.speed + accel * accel_time +
      std::normal_distribution<float>(0.0f, AGENT_SPEED_NOISE)(RngDet());
  next_agent.steer = steer;
  next_agent.position.x += std::normal_distribution<float>(0.0f, AGENT_POSITION_NOISE)(RngDet());
  next_agent.position.y += std::normal_distribution<float>(0.0f, AGENT_POSITION_NOISE)(RngDet());
  next_agent.heading.rotate(std::normal_distribution<float>(0.0f, AGENT_HEADING_NOISE)(RngDet()));

  return next_agent;
}
'''

if __name__ == '__main__':

    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with open('sync/camera.log', 'r') as f:
        lines = f.readlines()
    lines = [l.split(',') for l in lines]
    source_points = np.array([(float(l[0]), float(l[1])) for l in lines], dtype=np.float32)
    dest_points = np.array([(float(l[2]), float(l[3])) for l in lines], dtype=np.float32)
    cam_to_screen = cv2.findHomography(source_points, dest_points)[0]

    with open('sync/timings.log', 'r') as f:
        lines = f.readlines()
    timings = [float(l) for l in lines]

    with open('sync/ego_data.log', 'r') as f:
        lines = f.readlines()
    lines = [l.split(',') for l in lines]
    trajectory_points = np.array([(float(l[0]), float(l[1])) for l in lines], dtype=np.float32)[:-20]
    trajectory_headings = np.array([(float(l[2]), float(l[3])) for l in lines], dtype=np.float32)[:-20]
    trajectory_speeds = np.array([float(l[4]) for l in lines], dtype=np.float32)[:-20]

    with open('sync/params.log', 'r') as f:
        lines = f.readlines()
    trajectory_params = np.array([[x[0] for x in struct.iter_unpack('f', b64decode(l))] for l in lines], dtype=np.float32)

    video_in = cv2.VideoCapture('sync/video.mp4')
    has_first_frame = False
    episode_start_video_time = None
    previous_step_indexes = []
    subframe = None
    macro_actions = None
    encountered = []
    frame_index = -1
    while video_in.isOpened():
        (ret, frame) = video_in.read()
        if not ret:
            break
        frame_index += 1

        video_time = video_in.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if video_time == 0 and has_first_frame:
            break
        if not has_first_frame:
            has_first_frame = True
            first_frame_video_time = video_time

        episode_signal = frame[EPISODE_SIGNAL_POS[1]][EPISODE_SIGNAL_POS[0]]
        episode_signal = episode_signal[0] < 10 and episode_signal[1] < 10 and episode_signal[2] > 170 and episode_signal[2] < 190
        if episode_signal:

            if episode_start_video_time is None:
                episode_start_video_time = video_time

            OVERLAY_RANGE = 33
            OVERLAY_OFFSET = (720, 740)
            SCALE = 3.3

            # Prepare subframe
            step_index = [i for i in range(0, len(timings)) if timings[i] - timings[0] >= video_time - episode_start_video_time][0]
            step_pos = trajectory_points[step_index]


            if step_index % MACRO_LENGTH == 0 and step_index > 0 and step_index not in encountered:
                step_pos_transformed = np.rint(cv2.perspectiveTransform(step_pos.reshape(1, 1, 2), cam_to_screen).reshape(1, 2)[0]).astype(np.int32)
                subframe = frame[step_pos_transformed[1]-OVERLAY_RANGE:step_pos_transformed[1]+OVERLAY_RANGE,
                        step_pos_transformed[0]-OVERLAY_RANGE:step_pos_transformed[0]+OVERLAY_RANGE]
                subframe = cv2.resize(subframe, np.rint((subframe.shape[0] * SCALE, subframe.shape[0] * SCALE)).astype(np.int32))
                params = trajectory_params[int(step_index / MACRO_LENGTH) - 1]
                macro_actions = params.reshape(-1, 2)
                encountered.append(step_index)

            # Overlay subframe
            if True and subframe is not None:
                frame[OVERLAY_OFFSET[1]:OVERLAY_OFFSET[1]+subframe.shape[0],OVERLAY_OFFSET[0]:OVERLAY_OFFSET[0]+subframe.shape[1]] = subframe
                cv2.rectangle(frame,
                        (OVERLAY_OFFSET[0], OVERLAY_OFFSET[1]),
                        (OVERLAY_OFFSET[0] + subframe.shape[0], OVERLAY_OFFSET[1] + subframe.shape[1]),
                        (0, 0, 0), 5)

            # Draw macro-actions frame.
            if True and macro_actions is not None:

                # Handcrafted
                macro_actions = [
                    (1, -1), (1, 0), (1, 1),
                    (0.5, -0.5), (0.5, 0), (0.5, 0.5),
                    (0, 0)
                ]

                for (macro_action_index, macro_action_params) in enumerate([macro_actions[6]]):
                    speed = EGO_AGENT_SPEED_MAX * (macro_action_params[0] + 1) / 2.0
                    steer = EGO_AGENT_STEER_MAX * macro_action_params[1]
                    macro_action = [(speed, steer)]
                    for _ in range(0, int((MACRO_LENGTH - 1) / 2)):
                        macro_action.append((speed, steer))
                    for _ in range(int((MACRO_LENGTH - 1) / 2), MACRO_LENGTH - 1):
                        macro_action.append((speed, 0))

                    state = (trajectory_points[encountered[-1]], trajectory_headings[encountered[-1]], trajectory_speeds[encountered[-1]])
                    for action in macro_action:
                        next_state = actuate(*state, *action)
                        src_pos_transformed = np.rint(cv2.perspectiveTransform(state[0].reshape(1, 1, 2), cam_to_screen).reshape(1, 2)[0]).astype(np.int32)
                        src_pos_scaled = SCALE * (src_pos_transformed - step_pos_transformed)
                        src_pos_scaled[0] += OVERLAY_OFFSET[0] + subframe.shape[0] * 0.5
                        src_pos_scaled[1] += OVERLAY_OFFSET[1] + subframe.shape[1] * 0.5
                        src_pos_scaled = np.rint(src_pos_scaled).astype(np.int32)

                        dest_pos_transformed = np.rint(cv2.perspectiveTransform(next_state[0].reshape(1, 1, 2), cam_to_screen).reshape(1, 2)[0]).astype(np.int32)
                        dest_pos_scaled = SCALE * (dest_pos_transformed - step_pos_transformed)
                        dest_pos_scaled[0] += OVERLAY_OFFSET[0] + subframe.shape[0] * 0.5
                        dest_pos_scaled[1] += OVERLAY_OFFSET[1] + subframe.shape[1] * 0.5
                        dest_pos_scaled = np.rint(dest_pos_scaled).astype(np.int32)

                        cv2.line(frame, src_pos_scaled, dest_pos_scaled, COLORS[macro_action_index], 3)

                        state = next_state


            # Draw ego trajectory
            if True:
                trajectory_points_transformed = cv2.perspectiveTransform(trajectory_points.reshape(-1, 1, 2), cam_to_screen).reshape(-1, 2)
                for i in range(0, len(trajectory_points_transformed) - 1):
                    start = trajectory_points_transformed[i]
                    start = (int(start[0]), int(start[1]))
                    end = trajectory_points_transformed[i + 1]
                    end = (int(end[0]), int(end[1]))
                    cv2.line(frame, start, end, (0, 255, 0), 2)

            if frame_index == 148:
                while True:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(int(1000))


            # Draw search tree.
            if False and macro_actions is not None and frame_index == 148:

                # Handcrafted
                macro_actions = [
                    (1, -1), (1, 0), (1, 1),
                    (0.5, -0.5), (0.5, 0), (0.5, 0.5),
                    (0, 0)
                ]

                q = [((trajectory_points[encountered[-1]], trajectory_headings[encountered[-1]], trajectory_speeds[encountered[-1]]), 0)]
                while len(q) > 0:

                    # Breadth-first iteration stuff.
                    (state, depth) = q.pop(0)
                    if depth >= 1:
                        continue

                    CUSTOM_LENGTH = 3
                    for (macro_action_index, macro_action_params) in enumerate(macro_actions):

                        # Compute macro_action from params.
                        speed = EGO_AGENT_SPEED_MAX * (macro_action_params[0] + 1) / 2.0
                        steer = EGO_AGENT_STEER_MAX * macro_action_params[1]
                        macro_action = [(speed, steer)]
                        for _ in range(0, int((CUSTOM_LENGTH - 1) / 2)):
                            macro_action.append((speed, steer))
                        for _ in range(int((CUSTOM_LENGTH - 1) / 2), CUSTOM_LENGTH - 1):
                            macro_action.append((speed, 0))

                        # Step macro-action and draw path.
                        next_state = state
                        for action in macro_action:
                            next_state = actuate(*next_state, *action)
                            src_pos_transformed = np.rint(cv2.perspectiveTransform(state[0].reshape(1, 1, 2), cam_to_screen).reshape(1, 2)[0]).astype(np.int32)
                            dest_pos_transformed = np.rint(cv2.perspectiveTransform(next_state[0].reshape(1, 1, 2), cam_to_screen).reshape(1, 2)[0]).astype(np.int32)
                            #cv2.line(frame, src_pos_transformed, dest_pos_transformed, (39, 121, 233), 1) # COlor (39, 121, 233)

                        # Push to queue.
                        q.append((next_state, depth + 1))



    print(source_points, dest_points, cam_to_screen)
