#include "core/simulations/LightDark.h"

#include "core/Util.h"
#include <boost/functional/hash.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>
#include <iostream>

namespace simulations {

LightDark::Action LightDark::Action::Rand() {
  Action action{std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
  action.trigger = std::bernoulli_distribution(0.2)(Rng());
  return action;
}

uint64_t LightDark::Observation::Discretize() const {
  float grid_size = GOAL_ALLOWANCE / sqrtf(2);
  list_t<int> data {
    static_cast<int>(floorf((ego_agent_position.x + grid_size / 2) / grid_size)),
    static_cast<int>(floorf((ego_agent_position.y + grid_size / 2) / grid_size))
  };
  return boost::hash_value(data);
}

/* ====== Construction functions ====== */

LightDark::LightDark() : step(0), _is_terminal(false) {

}

LightDark LightDark::CreateRandom() {
  LightDark sim = SampleBeliefPrior();
  return sim;
}

/* ====== Belief related functions ======*/

LightDark LightDark::SampleBeliefPrior() {
  LightDark sim;
  sim.ego_agent_position.x = std::normal_distribution<float>(
      EGO_START.x, EGO_SPAWN_NOISE)(Rng());
  sim.ego_agent_position.y =std::normal_distribution<float>(
      EGO_START.y, EGO_SPAWN_NOISE)(Rng());
  return sim;
}

float LightDark::Error(const LightDark& other) const {
  return (ego_agent_position - other.ego_agent_position).norm();
}

/* ====== Bounds related functions ====== */
// Actually a high probability estimate, assuming actuation noise samples below 3 s.d.
float LightDark::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (GOAL - ego_agent_position).norm() - GOAL_ALLOWANCE);
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
std::tuple<LightDark, float, LightDark::Observation, float> LightDark::Step(
    const LightDark::Action& action, const LightDark::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  LightDark next_sim = *this;
  float reward;

  /* ====== Step 1: Update state.  ======*/
  if (action.trigger) {
    if ((next_sim.ego_agent_position - GOAL).norm() <= GOAL_ALLOWANCE) {
      reward = GOAL_REWARD;
    } else {
      reward = COLLISION_REWARD;
    }
    next_sim._is_terminal = true;
    next_sim.step++;
  } else {
    next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation);
    reward = STEP_REWARD;
    next_sim.step++;
    if (next_sim.step == MAX_STEPS) {
      if ((next_sim.ego_agent_position - GOAL).norm() <= GOAL_ALLOWANCE) {
        reward = GOAL_REWARD;
      } else {
        reward = COLLISION_REWARD;
      }
      next_sim._is_terminal = true;
    }
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }
  float log_prob = 0;

  float observation_noise = sqrtf(
      LIGHT_MIN_OBSERVATION_NOISE +
      LIGHT_OBSERVATION_NOISE_CONSTANT *
      (next_sim.ego_agent_position.x - LIGHT_POS) * (next_sim.ego_agent_position.x - LIGHT_POS));
  if (!observation) {
    new_observation.ego_agent_position = next_sim.ego_agent_position;
    new_observation.ego_agent_position.x += std::normal_distribution<float>(0.0, observation_noise)(RngDet());
    new_observation.ego_agent_position.y += std::normal_distribution<float>(0.0, observation_noise)(RngDet());
  }
  if constexpr (compute_log_prob) {
    log_prob += NormalLogProb(next_sim.ego_agent_position.x, observation_noise, new_observation.ego_agent_position.x);
    log_prob += NormalLogProb(next_sim.ego_agent_position.y, observation_noise, new_observation.ego_agent_position.y);
  }

  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<LightDark, float, LightDark::Observation, float> LightDark::Step<true>(
    const LightDark::Action& action, const LightDark::Observation* observation) const;
template std::tuple<LightDark, float, LightDark::Observation, float> LightDark::Step<false>(
    const LightDark::Action& action, const LightDark::Observation* observation) const;

/* ====== Serialization functions ====== */

void LightDark::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
}

cv::Mat LightDark::Render(const list_t<LightDark>& belief_sims) const {

  constexpr float SCENARIO_MIN = -7.0f;
  constexpr float SCENARIO_MAX = 7.0f;
  constexpr float RESOLUTION = 0.02f;
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

  double lowest_intensity = std::numeric_limits<double>::infinity();
  double highest_intensity = -std::numeric_limits<double>::infinity();
  for (int x = 0; x < frame.cols; x++) {
    double intensity = std::abs(
        (static_cast<float>(x) * RESOLUTION + static_cast<float>(SCENARIO_MIN))
        - static_cast<float>(LIGHT_POS));
    lowest_intensity = std::min(lowest_intensity, intensity);
    highest_intensity = std::max(highest_intensity, intensity);
  }
  for (int x = 0; x < frame.cols; x++) {
    double intensity = std::abs(
        (static_cast<float>(x) * RESOLUTION + static_cast<float>(SCENARIO_MIN))
        - static_cast<float>(LIGHT_POS));
    intensity = 255 * (1 - (intensity - lowest_intensity) / (highest_intensity - lowest_intensity));
    frame.col(x).setTo(cv::Scalar(intensity, intensity, intensity));
  }

  cv::drawMarker(frame, to_frame(EGO_START),
      cv::Scalar(255, 255, 0), cv::MARKER_TILTED_CROSS, 40, 3, cv::LINE_AA);
  cv::circle(frame, to_frame(GOAL), to_frame_dist(EGO_RADIUS),
      cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

  // Draw ego agent.
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  for (const simulations::LightDark& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.ego_agent_position), cv::Scalar(0, 255, 255),
        cv::MARKER_CROSS, 3, 2, cv::LINE_4);
  }

  return frame;
}

}
