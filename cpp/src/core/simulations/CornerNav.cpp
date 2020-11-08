#include "core/simulations/CornerNav.h"

#include "core/Util.h"
#include "rvo2/RVO.h"
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>

namespace simulations {

const boost::geometry::model::ring<vector_t, true, false> REGION_INNER_EXO = {
    vector_t{-12.5f + CornerNav::EXO_RADIUS, 20 - CornerNav::EXO_RADIUS},
    vector_t{12.5f - CornerNav::EXO_RADIUS, 20 - CornerNav::EXO_RADIUS},
    vector_t{12.5f - CornerNav::EXO_RADIUS, -20 + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, -20 + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, -7.5f - CornerNav::EXO_RADIUS},
    vector_t{0 + CornerNav::EXO_RADIUS, -7.5f - CornerNav::EXO_RADIUS},
    vector_t{0 + CornerNav::EXO_RADIUS, 7.5f + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, 7.5f + CornerNav::EXO_RADIUS}
};

const boost::geometry::model::linestring<vector_t> REGION_INNER_EXO_PERIMETER = {
    vector_t{-12.5f + CornerNav::EXO_RADIUS, 20 - CornerNav::EXO_RADIUS},
    vector_t{12.5f - CornerNav::EXO_RADIUS, 20 - CornerNav::EXO_RADIUS},
    vector_t{12.5f - CornerNav::EXO_RADIUS, -20 + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, -20 + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, -7.5f - CornerNav::EXO_RADIUS},
    vector_t{0 + CornerNav::EXO_RADIUS, -7.5f - CornerNav::EXO_RADIUS},
    vector_t{0 + CornerNav::EXO_RADIUS, 7.5f + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, 7.5f + CornerNav::EXO_RADIUS},
    vector_t{-12.5f + CornerNav::EXO_RADIUS, 20 - CornerNav::EXO_RADIUS}
};

const boost::geometry::model::ring<vector_t, true, false> REGION_INNER_EGO = {
    vector_t{-12.5f + CornerNav::EGO_RADIUS, 20 - CornerNav::EGO_RADIUS},
    vector_t{12.5f - CornerNav::EGO_RADIUS, 20 - CornerNav::EGO_RADIUS},
    vector_t{12.5f - CornerNav::EGO_RADIUS, -20 + CornerNav::EGO_RADIUS},
    vector_t{-12.5f + CornerNav::EGO_RADIUS, -20 + CornerNav::EGO_RADIUS},
    vector_t{-12.5f + CornerNav::EGO_RADIUS, -7.5f - CornerNav::EGO_RADIUS},
    vector_t{0 + CornerNav::EGO_RADIUS, -7.5f - CornerNav::EGO_RADIUS},
    vector_t{0 + CornerNav::EGO_RADIUS, 7.5f + CornerNav::EGO_RADIUS},
    vector_t{-12.5f + CornerNav::EGO_RADIUS, 7.5f + CornerNav::EGO_RADIUS}
};

CornerNav::Action CornerNav::Action::Rand() {
  return {std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
}

uint64_t CornerNav::Observation::Discretize() const {
  array_t<int, 2 + 2 * NUM_EXO_AGENTS> data;
  data[0] = static_cast<int>(floorf(ego_agent_position.x / 1.0f));
  data[1] = static_cast<int>(floorf(ego_agent_position.y / 1.0f));
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    data[2 + 2 * i + 0] = static_cast<int>(floorf(exo_agent_positions[i].x / 1.0f));
    data[2 + 2 * i + 1] = static_cast<int>(floorf(exo_agent_positions[i].y / 1.0f));
  };
  return boost::hash_value(data);
}

/* ====== Construction functions ====== */

CornerNav::CornerNav() : step(0), _is_terminal(false) {

}

CornerNav CornerNav::CreateRandom() {

  CornerNav sim = SampleBeliefPrior();

  // Reset RVOs.
  if (_rvo) {
    delete _rvo;
  }
  _rvo = new RVO::RVOSimulator();

  // Initialize RVO.
  _rvo->setTimeStep(static_cast<float>(DELTA));
  _rvo->processObstacles();

  // Add exo agents.
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    _rvo_ids[i] = _rvo->addAgent(
        ToRVO(sim.exo_agent_positions[i]), // Position.
        8.0, // Neighbour distance.
        10, // Max neighbours.
        static_cast<float>(DELTA), // Time horizon.
        static_cast<float>(DELTA), // Time horizon (obstacles).
        static_cast<float>(EXO_RADIUS),
        static_cast<float>(EXO_SPEED), // Max speed.
        ToRVO(sim.exo_agent_previous_velocities[i]));
  }

  return sim;
}

/* ====== Belief related functions ======*/

CornerNav CornerNav::SampleBeliefPrior() {

  CornerNav sim;

  // Create ego agent.
  do {
    sim.ego_agent_position = EGO_AGENT_START;
    sim.ego_agent_position.x += std::normal_distribution<float>(0.0f, EGO_SPAWN_NOISE)(RngDet());
    sim.ego_agent_position.y += std::normal_distribution<float>(0.0f, EGO_SPAWN_NOISE)(RngDet());
  } while (!boost::geometry::within(sim.ego_agent_position, REGION_INNER_EGO));

  // Create exo agents.
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    do {
      sim.exo_agent_positions[i] = EXO_AGENT_START[i];
      sim.exo_agent_positions[i].x += std::normal_distribution<float>(0.0f, EXO_SPAWN_NOISE)(RngDet());
      sim.exo_agent_positions[i].y += std::normal_distribution<float>(0.0f, EXO_SPAWN_NOISE)(RngDet());
    } while (!boost::geometry::within(sim.ego_agent_position, REGION_INNER_EGO));
    sim.exo_agent_target_velocities[i] = vector_t(EXO_SPEED, 0).rotated(std::uniform_real_distribution<float>(-PI, PI)(Rng()));
    sim.exo_agent_previous_velocities[i] = vector_t(0, 0);
  }

  return sim;
}

float CornerNav::Error(const CornerNav& other) const {
  float error = 0.0;
  error += (ego_agent_position - other.ego_agent_position).norm();
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    error += (exo_agent_positions[i] - other.exo_agent_positions[i]).norm();
  }
  return error / static_cast<float>(1 + NUM_EXO_AGENTS);
}

/* ====== Bounds related functions ====== */
// Actually a high probability estimate, assuming actuation noise samples below 3 s.d.
float CornerNav::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (GOAL - ego_agent_position).norm() - EGO_RADIUS - GOAL_RADIUS);
  float max_distance_per_step = EGO_SPEED * DELTA;
  size_t steps = static_cast<size_t>(round(ceilf(distance / max_distance_per_step)));
  if (steps <= 1) {
    return GOAL_REWARD;
  } else {
    return (1 - powf(GAMMA, static_cast<float>(steps) - 1)) / (1 - static_cast<float>(steps)) * STEP_REWARD +
      powf(GAMMA, static_cast<float>(steps) - 1) * GOAL_REWARD;
  }
}

/* ====== Stepping functions ====== */

template <bool compute_log_prob>
std::tuple<CornerNav, float, CornerNav::Observation, float> CornerNav::Step(
    const CornerNav::Action& action, const CornerNav::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  CornerNav next_sim = *this;
  float reward;


  /* ====== Step 1: Update state.  ======*/

  // Add exo agents.
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    _rvo->setAgentPosition(_rvo_ids[i], ToRVO(exo_agent_positions[i]));
    _rvo->setAgentVelocity(_rvo_ids[i], ToRVO(exo_agent_previous_velocities[i]));
    _rvo->setAgentPrefVelocity(_rvo_ids[i], ToRVO(exo_agent_target_velocities[i]));
  }
  // Update exo agents.
  _rvo->doStep();
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    next_sim.exo_agent_previous_velocities[i] = FromRVO(_rvo->getAgentVelocity(_rvo_ids[i]));
    next_sim.exo_agent_positions[i] += DELTA * next_sim.exo_agent_previous_velocities[i];
    next_sim.exo_agent_positions[i].x += std::normal_distribution<float>(0.0f, EXO_ACTUATION_NOISE)(RngDet());
    next_sim.exo_agent_positions[i].y += std::normal_distribution<float>(0.0f, EXO_ACTUATION_NOISE)(RngDet());

    if (!boost::geometry::within(next_sim.exo_agent_positions[i], REGION_INNER_EXO)) {
      boost::geometry::model::linestring<vector_t> segment{
            exo_agent_positions[i],
            next_sim.exo_agent_positions[i]
      };
      list_t<vector_t> intersections;
      boost::geometry::intersection(segment, REGION_INNER_EXO_PERIMETER, intersections);
      if (intersections.size() > 0) {
        next_sim.exo_agent_positions[i] = intersections[0];
        while (true) {
          vector_t next_vel = vector_t(EXO_SPEED, 0).rotated(std::uniform_real_distribution<float>(-PI, PI)(RngDet()));
          if (boost::geometry::within(next_sim.exo_agent_positions[i] + DELTA * next_vel, REGION_INNER_EXO)) {
            next_sim.exo_agent_target_velocities[i] = next_vel;
            break;
          }
        }
      }
    }
  }

  // Update ego agent.
  next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation);
  next_sim.ego_agent_position.x += std::normal_distribution<float>(0.0f, EGO_ACTUATION_NOISE)(RngDet());
  next_sim.ego_agent_position.y += std::normal_distribution<float>(0.0f, EGO_ACTUATION_NOISE)(RngDet());

  next_sim.step++;

  // Check terminal and calculate rewards.
  if (!next_sim._is_terminal) {
    if (!boost::geometry::within(next_sim.ego_agent_position, REGION_INNER_EGO)) {
      next_sim._is_terminal = true;
      reward = COLLISION_REWARD;
    }
  }
  if (!next_sim._is_terminal) {
    for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
      if ((next_sim.exo_agent_positions[i] - next_sim.ego_agent_position).norm() <= EGO_RADIUS + EXO_RADIUS) {
        next_sim._is_terminal = true;
        reward = COLLISION_REWARD;
        break;
      }
    }
  }
  if (!next_sim._is_terminal) {
    if ((next_sim.ego_agent_position - GOAL).norm() <= EGO_RADIUS + GOAL_RADIUS) {
      next_sim._is_terminal = true;
      reward = GOAL_REWARD;
    }
  }
  if (!next_sim._is_terminal) {
    if (next_sim.step == MAX_STEPS) {
      reward = COLLISION_REWARD;
      next_sim._is_terminal = true;
    }
  }
  if (!next_sim._is_terminal) {
    reward = STEP_REWARD;
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }
  float log_prob = 0;

  if (!observation) {
    new_observation.ego_agent_position = next_sim.ego_agent_position;
    new_observation.ego_agent_position.x += std::normal_distribution<float>(0.0, EGO_OBSERVATION_NOISE)(RngDet());
    new_observation.ego_agent_position.y += std::normal_distribution<float>(0.0, EGO_OBSERVATION_NOISE)(RngDet());
  }
  if constexpr (compute_log_prob) {
    log_prob += NormalLogProb(next_sim.ego_agent_position.x, EGO_OBSERVATION_NOISE, new_observation.ego_agent_position.x);
    log_prob += NormalLogProb(next_sim.ego_agent_position.y, EGO_OBSERVATION_NOISE, new_observation.ego_agent_position.y);
  }

  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    if (!observation) {
      new_observation.exo_agent_positions[i] = next_sim.exo_agent_positions[i];
      new_observation.exo_agent_positions[i].x += std::normal_distribution<float>(0.0, EXO_OBSERVATION_NOISE)(RngDet());
      new_observation.exo_agent_positions[i].y += std::normal_distribution<float>(0.0, EXO_OBSERVATION_NOISE)(RngDet());
    }
    if constexpr (compute_log_prob) {
      log_prob += NormalLogProb(next_sim.exo_agent_positions[i].x, EXO_OBSERVATION_NOISE, new_observation.exo_agent_positions[i].x);
      log_prob += NormalLogProb(next_sim.exo_agent_positions[i].y, EXO_OBSERVATION_NOISE, new_observation.exo_agent_positions[i].y);
    }
  }

  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<CornerNav, float, CornerNav::Observation, float> CornerNav::Step<true>(
    const CornerNav::Action& action, const CornerNav::Observation* observation) const;
template std::tuple<CornerNav, float, CornerNav::Observation, float> CornerNav::Step<false>(
    const CornerNav::Action& action, const CornerNav::Observation* observation) const;

/* ====== Serialization functions ====== */

void CornerNav::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    exo_agent_positions[i].Encode(data);
    exo_agent_target_velocities[i].Encode(data);
    exo_agent_previous_velocities[i].Encode(data);
  }
}

cv::Mat CornerNav::Render(const list_t<CornerNav>& belief_sims) const {

  constexpr float SCENARIO_MIN = -22.0f;
  constexpr float SCENARIO_MAX = 22.0f;
  constexpr float RESOLUTION = 0.05f;
  auto to_frame = [&](const vector_t& vector) {
    return cv::Point{
      static_cast<int>((vector.x - SCENARIO_MIN) / RESOLUTION),
      static_cast<int>((SCENARIO_MAX - vector.y) / RESOLUTION)
    };
  };
  auto to_frame_dist = [&](float d) {
    return static_cast<int>(d / RESOLUTION);
  };

  cv::Mat frame(
      static_cast<int>((SCENARIO_MAX - SCENARIO_MIN) / RESOLUTION),
      static_cast<int>((SCENARIO_MAX - SCENARIO_MIN) / RESOLUTION),
      CV_8UC3,
      cv::Scalar(255, 255, 255));


  // Draw region.
  for (size_t i = 0; i < REGION.size(); i++) {
    cv::line(frame,
        to_frame(REGION[i]),
        to_frame(REGION[(i + 1) % REGION.size()]),
        cv::Scalar(0, 0, 0),
        5, cv::LINE_AA);
  }

  // Draw exo agents.
  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    cv::circle(frame, to_frame(exo_agent_positions[i]), to_frame_dist(EXO_RADIUS),
        cv::Scalar(42, 209, 255), -1, cv::LINE_AA);
    for (const simulations::CornerNav& belief_sim : belief_sims) {
      cv::drawMarker(frame, to_frame(belief_sim.exo_agent_positions[i]), cv::Scalar(0, 0, 255),
          cv::MARKER_CROSS, 2, 1, cv::LINE_4);
    }
  }

  // Draw ego agent.
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
  for (const simulations::CornerNav& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.ego_agent_position), cv::Scalar(0, 0, 255),
        cv::MARKER_CROSS, 2, 1, cv::LINE_4);
  }

  cv::circle(frame, to_frame(GOAL), to_frame_dist(GOAL_RADIUS),
      cv::Scalar(255, 0, 255), -1, cv::LINE_AA);

  return frame;


}

}
