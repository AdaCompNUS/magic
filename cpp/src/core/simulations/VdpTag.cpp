#include "core/simulations/VdpTag.h"

#include "core/Util.h"
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

namespace simulations {

VdpTag::Action VdpTag::Action::Rand() {
  Action action(std::uniform_real_distribution<float>(0, 2 * PI)(Rng()));
  action.look = std::bernoulli_distribution(0.5)(Rng());
  return action;
}

list_t<list_t<VdpTag::Action>> VdpTag::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 8; i++) {
    macro_actions.emplace_back();
    for (size_t j = 0; j < length; j++) {
      macro_actions.back().push_back({static_cast<float>(i) * 2 * PI / 8});
    }
  }
  Action trigger_action;
  trigger_action.look = true;
  macro_actions.emplace_back();
  macro_actions.back().emplace_back(trigger_action);
  return macro_actions;
}

/* ====== Construction related functions ====== */

VdpTag::VdpTag() : _is_terminal(false) {

}

VdpTag VdpTag::CreateRandom() {
  return SampleBeliefPrior();
}

/* ====== Belief related functions ====== */
VdpTag VdpTag::SampleBeliefPrior() {
  VdpTag sim;
  sim.ego_agent_position = vector_t(0.0f, 0.0f);
  sim.exo_agent_position.x = std::uniform_real_distribution<float>(-4.0, 4.0)(Rng());
  sim.exo_agent_position.y = std::uniform_real_distribution<float>(-4.0, 4.0)(Rng());
  return sim;
}

float VdpTag::Error(const VdpTag& other) const {
  float error = 0;
  error += (ego_agent_position - other.ego_agent_position).norm();
  error += (exo_agent_position - other.exo_agent_position).norm();
  return error / 2;
}

/* ====== Bounds related functions ====== */
float VdpTag::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (exo_agent_position - ego_agent_position).norm() - TAG_RADIUS);
  float max_distance_per_step = AGENT_SPEED * DELTA + 4.0f * DELTA + 3 * POS_STD ;
  size_t steps = static_cast<size_t>(round(ceilf(distance / max_distance_per_step)));
  if (steps <= 1) {
    return TAG_REWARD;
  } else {
    return (1 - powf(GAMMA, static_cast<float>(steps) - 1)) / (1 - static_cast<float>(steps)) * STEP_REWARD +
      powf(GAMMA, static_cast<float>(steps) - 1) * TAG_REWARD;
  }
}

/* ====== Stepping functions ====== */

float VdpTag::Cross(const vector_t& a, const vector_t& b) {
  return a.x * b.y - b.x * a.y;
}

vector_t VdpTag::VdpDynamics(const vector_t& v) const {
  return {
    MU * (v.x - v.x * v.x * v.x / 3 - v.y),
    v.x / MU
  };
}

vector_t VdpTag::Rk4Step(const vector_t& v) const {
  float h = RK4_STEP_SIZE;
  vector_t k1 = VdpDynamics(v);
  vector_t k2 = VdpDynamics(v + k1 * h / 2);
  vector_t k3 = VdpDynamics(v + k2 * h / 2);
  vector_t k4 = VdpDynamics(v + k3 * h);
  return v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

vector_t VdpTag::BarrierStop(const vector_t& v, const vector_t& d) const {
  float shortest_u = 1.0f + 2 * std::numeric_limits<float>::epsilon();
  vector_t q = v;
  vector_t s = d;

  for (vector_t dir : list_t<vector_t>{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}) {
    vector_t p = 0.2f * dir;
    vector_t r = 2.8f * dir;
    float rxs = Cross(r, s);
    if (rxs == 0.0f) {
      continue;
    } else {
      vector_t qmp = q - p;
      float u = Cross(qmp, r) / rxs;
      float t = Cross(qmp, s) / rxs;
      if (0.0f <= u && u < shortest_u && 0.0f <= t && t <= 1.0f) {
        shortest_u = u;
      }
    }
  }

  return v + (shortest_u - 2 * std::numeric_limits<float>::epsilon()) * d;
}

size_t VdpTag::ActiveBeam(const vector_t& v) const {
  float angle = AngleTo(vector_t(1.0f, 0.0f), v);
  while (angle <= 0.0f) {
    angle += 2 * PI;
  }
  size_t x = static_cast<size_t>(lround(ceilf(8 * angle / (2 * PI))) - 1);
  return std::max(static_cast<size_t>(0), std::min(static_cast<size_t>(7), x));
}

template <bool compute_log_prob>
std::tuple<VdpTag, float, VdpTag::Observation, float> VdpTag::Step(
    const VdpTag::Action& action, const VdpTag::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  VdpTag next_sim = *this;
  float reward = 0;

  /* ====== Step 1: Update state. ====== */
  next_sim.ego_agent_position = BarrierStop(
      next_sim.ego_agent_position,
      AGENT_SPEED * DELTA * vector_t(1, 0).rotated(action.angle));

  for (size_t i = 0; i < RK4_STEP_ITER; i++) {
    next_sim.exo_agent_position = Rk4Step(next_sim.exo_agent_position);
  }
  next_sim.exo_agent_position.x += std::normal_distribution<float>(0.0, POS_STD)(RngDet());
  next_sim.exo_agent_position.y += std::normal_distribution<float>(0.0, POS_STD)(RngDet());
  next_sim.step++;

  // Check terminal and rewards.
  if ((next_sim.ego_agent_position - next_sim.exo_agent_position).norm() < TAG_RADIUS) {
    reward = TAG_REWARD;
    next_sim._is_terminal = true;
  } else {
    reward = STEP_REWARD;
  }
  if (action.look) {
    reward += ACTIVE_MEAS_REWARD;
  }

  if (!_is_terminal) {
    if (next_sim.step == MAX_STEPS) {
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation= *observation;
  }
  float log_prob = 0;
  vector_t rel_pos = next_sim.exo_agent_position - next_sim.ego_agent_position;
  float dist = rel_pos.norm();
  size_t active_beam = ActiveBeam(rel_pos);
  if (action.look) {
    if (!observation) {
      new_observation.beam_distances[active_beam] = std::normal_distribution<float>(dist, ACTIVE_MEAS_STD)(RngDet());
    }
    if constexpr (compute_log_prob) {
      log_prob += NormalLogProb(dist, ACTIVE_MEAS_STD, new_observation.beam_distances[active_beam]);
    }
  } else {
    if (!observation) {
      new_observation.beam_distances[active_beam] = std::normal_distribution<float>(dist, MEAS_STD)(RngDet());
    }
    if constexpr (compute_log_prob) {
      log_prob += NormalLogProb(dist, MEAS_STD, new_observation.beam_distances[active_beam]);
    }
  }
  for (size_t i = 0; i < new_observation.beam_distances.size(); i++) {
    if (i != active_beam) {
      if (!observation) {
        new_observation.beam_distances[i] = std::normal_distribution<float>(1.0, MEAS_STD)(RngDet());
      }
      if constexpr (compute_log_prob) {
        log_prob += NormalLogProb(1.0, MEAS_STD, new_observation.beam_distances[i]);
      }
    }
  }

  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<VdpTag, float, VdpTag::Observation, float> VdpTag::Step<true>(
    const VdpTag::Action& action, const VdpTag::Observation* observation) const;
template std::tuple<VdpTag, float, VdpTag::Observation, float> VdpTag::Step<false>(
    const VdpTag::Action& action, const VdpTag::Observation* observation) const;

/* ====== Serialization functions ====== */

void VdpTag::Encode(list_t<float>& data) const {
  ego_agent_position.Encode(data);
  exo_agent_position.Encode(data);
}

cv::Mat VdpTag::Render(const list_t<VdpTag>& belief_sims) const {

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

  for (vector_t dir : list_t<vector_t>{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}) {
    cv::line(
        frame,
        to_frame(0.2f * dir),
        to_frame(3.0f * dir), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }

  cv::circle(frame, to_frame(exo_agent_position), to_frame_dist(TAG_RADIUS),
      cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
  for (const simulations::VdpTag& sim : belief_sims) {
    cv::drawMarker(frame, to_frame(sim.exo_agent_position),
        cv::Scalar(0, 0, 0), cv::MARKER_TILTED_CROSS, static_cast<int>(to_frame_dist(VdpTag::TAG_RADIUS) * 0.2f), 2, cv::LINE_AA);
  }

  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(TAG_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  for (vector_t dir : list_t<vector_t>{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}) {
    cv::line(
        frame,
        to_frame(0.2f * dir),
        to_frame(3.0f * dir), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  }

  return frame;

}


}
