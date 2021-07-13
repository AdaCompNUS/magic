#include "core/simulations/LightDark.h"

#include "core/Util.h"
#include <boost/functional/hash.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <random>
#include <stdexcept>

namespace simulations {

LightDark::Action LightDark::Action::Rand() {
  Action action{std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
  action.trigger = std::bernoulli_distribution(0.2)(Rng());
  return action;
}

uint64_t LightDark::Observation::Discretize() const {
  float grid_size = DELTA * EGO_SPEED;
  list_t<int> data {
    static_cast<int>(floorf(ego_agent_position.x / grid_size)),
    static_cast<int>(floorf(ego_agent_position.y / grid_size))
  };
  return boost::hash_value(data);
}

list_t<list_t<LightDark::Action>> LightDark::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 8; i++) {
    macro_actions.emplace_back();
    for (size_t j = 0; j < length; j++) {
      macro_actions.back().push_back({static_cast<float>(i) * 2 * PI / 8});
    }
  }
  Action trigger_action;
  trigger_action.trigger = true;
  macro_actions.emplace_back();
  macro_actions.back().emplace_back(trigger_action);
  return macro_actions;
}

list_t<list_t<LightDark::Action>> LightDark::Action::Deserialize(const list_t<float>& params, size_t macro_length) {
  list_t<list_t<LightDark::Action>> macro_actions = StandardMacroActionDeserialization<LightDark::Action>(params, macro_length);
  Action trigger_action;
  trigger_action.trigger = true;
  macro_actions.emplace_back();
  macro_actions.back().emplace_back(trigger_action);
  return macro_actions;
}

/* ====== Construction functions ====== */

LightDark::LightDark() : step(0), _is_terminal(false), _is_failure(false) {

}

LightDark LightDark::CreateRandom() {

  do {
    EGO_START_MEAN.x = std::uniform_real_distribution<float>(RANDOMIZATION_REGION[0].x, RANDOMIZATION_REGION[2].x)(Rng());
    EGO_START_MEAN.y = std::uniform_real_distribution<float>(RANDOMIZATION_REGION[0].y, RANDOMIZATION_REGION[2].y)(Rng());
    GOAL.x = std::uniform_real_distribution<float>(RANDOMIZATION_REGION[0].x, RANDOMIZATION_REGION[2].x)(Rng());
    GOAL.y = std::uniform_real_distribution<float>(RANDOMIZATION_REGION[0].y, RANDOMIZATION_REGION[2].y)(Rng());
    LIGHT_POS = std::uniform_real_distribution<float>(RANDOMIZATION_REGION[0].x, RANDOMIZATION_REGION[2].x)(Rng());
  } while ((EGO_START_MEAN - GOAL).norm() < 4 || (std::abs(EGO_START_MEAN.x - LIGHT_POS) < 4 || (std::abs(GOAL.x - LIGHT_POS) < 4)));

  /*
  EGO_START_MEAN = {-2.0f, -2.5f};
  GOAL = {-3.5f, 2.9f};
  LIGHT_POS = 3.5f;
  */

  return SampleBeliefPrior();
}

/* ====== Belief related functions ======*/

LightDark LightDark::SampleBeliefPrior() {
  LightDark sim;
  sim.ego_agent_position.x = std::normal_distribution<float>(
      EGO_START_MEAN.x, EGO_START_STD)(Rng());
  sim.ego_agent_position.y =std::normal_distribution<float>(
      EGO_START_MEAN.y, EGO_START_STD)(Rng());
  return sim;
}

float LightDark::Error(const LightDark& other) const {
  return (ego_agent_position - other.ego_agent_position).norm();
}

/* ====== Bounds related functions ====== */
// Actually a high probability estimate, assuming actuation noise samples below 3 s.d.
float LightDark::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (GOAL - ego_agent_position).norm() - EGO_RADIUS);
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
    if ((next_sim.ego_agent_position - GOAL).norm() <= EGO_RADIUS) {
      reward = GOAL_REWARD;
      next_sim._is_failure = false;
    } else {
      reward = COLLISION_REWARD;
      next_sim._is_failure = true;
    }
    next_sim._is_terminal = true;
    next_sim.step++;
  } else {
    next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation);
    reward = STEP_REWARD;
    next_sim.step++;
    if (next_sim.step == MAX_STEPS) {
      reward = COLLISION_REWARD;
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }
  float log_prob = 0;

  bool in_light = abs(next_sim.ego_agent_position.x - LIGHT_POS) <= LIGHT_WIDTH;
  if (!observation) {
    if (in_light) {
      new_observation.ego_agent_position = next_sim.ego_agent_position;
      new_observation.ego_agent_position.x += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
      new_observation.ego_agent_position.y += std::normal_distribution<float>(0.0, OBSERVATION_NOISE)(RngDet());
    } else {
      new_observation.ego_agent_position.x = std::numeric_limits<float>::quiet_NaN();
      new_observation.ego_agent_position.y = std::numeric_limits<float>::quiet_NaN();
    }
  }
  if constexpr (compute_log_prob) {
    if (in_light) {
      if (std::isnan(new_observation.ego_agent_position.x) && std::isnan(new_observation.ego_agent_position.y)) {
        log_prob += -std::numeric_limits<float>::infinity();
      } else {
        log_prob += NormalLogProb(next_sim.ego_agent_position.x, OBSERVATION_NOISE, new_observation.ego_agent_position.x);
        log_prob += NormalLogProb(next_sim.ego_agent_position.y, OBSERVATION_NOISE, new_observation.ego_agent_position.y);
      }
    } else {
      if (std::isnan(new_observation.ego_agent_position.x) && std::isnan(new_observation.ego_agent_position.y)) {
        log_prob += 0;
      } else {
        log_prob += -std::numeric_limits<float>::infinity();
      }
    }
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

void LightDark::EncodeContext(list_t<float>& data) {
  GOAL.Encode(data);
  data.emplace_back(LIGHT_POS);
}

cv::Mat LightDark::Render(const list_t<LightDark>& belief_sims,
    const list_t<list_t<Action>>& macro_actions, const vector_t& macro_action_start) const {

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

  for (int x = 0; x < frame.cols; x++) {
    float x_world = static_cast<float>(x) * RESOLUTION + SCENARIO_MIN;
    float intensity = abs(x_world - LIGHT_POS) <= LIGHT_WIDTH ? 255 : 128;
    frame.col(x).setTo(cv::Scalar(intensity, intensity, intensity));
  }

  cv::drawMarker(frame, to_frame(EGO_START_MEAN),
      cv::Scalar(255, 255, 0), cv::MARKER_TILTED_CROSS, 30, 2, cv::LINE_AA);
  cv::drawMarker(frame, to_frame(GOAL),
      cv::Scalar(0, 255, 0), cv::MARKER_TILTED_CROSS, 30, 2, cv::LINE_AA);

  for (const simulations::LightDark& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.ego_agent_position), cv::Scalar(0, 255, 255),
        cv::MARKER_CROSS, 3, 2, cv::LINE_4);
  }

  // Draw ego agent.
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  const static list_t<cv::Scalar> colors = {
    {75, 25, 230},
    {49, 130, 245},
    {25, 225, 255},
    {240, 240, 70},
    {75, 180, 60},
    {180, 30, 145},
    {230, 50, 240},
    {216, 99, 67}
  };

  for (size_t i = 0; i < macro_actions.size(); i++) {
    vector_t s = macro_action_start;
    for (const Action& a : macro_actions[i]) {
      vector_t e = s + vector_t(DELTA * EGO_SPEED, 0).rotated(a.orientation);
      if (!a.trigger) {
        cv::line(frame, to_frame(s), to_frame(e),
            colors[i], 5, cv::LINE_AA);
      }
      s = e;
    }
  }

  if (_is_terminal) {
    if ((ego_agent_position - GOAL).norm() <= EGO_RADIUS) {
      cv::putText(frame,
          "Stop (Success)",
          to_frame(ego_agent_position + vector_t(1.0, - EGO_RADIUS / 2)),
          cv::FONT_HERSHEY_DUPLEX,
          1.0,
          cv::Scalar(0, 255, 0),
          2,
          cv::LINE_AA);
    } else {
      cv::putText(frame,
          "Stop (Failure)",
          to_frame(ego_agent_position + vector_t(1.0, - EGO_RADIUS / 2)),
          cv::FONT_HERSHEY_DUPLEX,
          1.0,
          cv::Scalar(0, 0, 255),
          2,
          cv::LINE_AA);
    }
  }


  return frame;
}

}

