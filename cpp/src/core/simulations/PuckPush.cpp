#include "core/simulations/PuckPush.h"

#include "core/Util.h"
#include <boost/math/tools/roots.hpp>
#include <functional>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <random>

namespace simulations {

const boost::geometry::model::ring<vector_t, true, false> BOARD_REGION_BOT_BOOST(
  PuckPush::BOARD_REGION_BOT.begin(), PuckPush::BOARD_REGION_BOT.end());

const boost::geometry::model::ring<vector_t, true, false> BOARD_REGION_PUCK_BOOST(
  PuckPush::BOARD_REGION_PUCK.begin(), PuckPush::BOARD_REGION_PUCK.end());

PuckPush::Action PuckPush::Action::Rand() {
  return {std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
}

list_t<list_t<PuckPush::Action>> PuckPush::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 8; i++) {
    macro_actions.emplace_back();
    for (size_t j = 0; j < length; j++) {
      macro_actions.back().push_back({static_cast<float>(i) * 2 * PI / 8});
    }
  }
  return macro_actions;
}

list_t<list_t<PuckPush::Action>> PuckPush::Action::Deserialize(const list_t<float>& params, size_t macro_length) {
  return StandardMacroActionDeserialization<PuckPush::Action>(params, macro_length);
}

uint64_t PuckPush::Observation::Discretize() const {
  list_t<int> data {
    static_cast<int>(floorf(bot_position.x / 1.0f)),
    static_cast<int>(floorf(bot_position.y / 1.0f)),
    static_cast<int>(floorf(puck_position.x / 1.0f)),
    static_cast<int>(floorf(puck_position.y / 1.0f))
  };
  return boost::hash_value(data);
}

PuckPush::PuckPush() : step(0), _is_terminal(false), _is_failure(false) {

}

PuckPush PuckPush::CreateRandom() {

  if constexpr (RANDOMIZE_GOAL) {
    GOAL.x = std::uniform_real_distribution<float>(GOAL_REGION[0].x, GOAL_REGION[2].x)(Rng());
    GOAL.y = std::uniform_real_distribution<float>(GOAL_REGION[0].y, GOAL_REGION[2].y)(Rng());
  } else {
    GOAL = {1058, 428};
  }

  return SampleBeliefPrior();
}

/* ====== Belief related functions ====== */
PuckPush PuckPush::SampleBeliefPrior() {
  PuckPush sim;
  sim.bot_position = BOT_START_POSITION;
  sim.bot_position.x += std::normal_distribution<float>(0.0, INITIAL_POSITION_NOISE)(Rng());
  sim.bot_position.y += std::normal_distribution<float>(0.0, INITIAL_POSITION_NOISE)(Rng());
  sim.puck_position = PUCK_START_POSITION;
  sim.puck_position.x += std::normal_distribution<float>(0.0, INITIAL_POSITION_NOISE)(Rng());
  sim.puck_position.y += std::normal_distribution<float>(0.0, INITIAL_POSITION_NOISE)(Rng());
  return sim;
}

float PuckPush::Error(const PuckPush& other) const {
  float error = 0;
  error += (bot_position - other.bot_position).norm();
  error += (puck_position - other.puck_position).norm();
  return error / 2;
}

/* ====== Bounds related functions ====== */
float PuckPush::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.
  float distance = (GOAL - puck_position).norm() - (GOAL_RADIUS - PUCK_RADIUS);
  float max_distance_per_step = BOT_SPEED * DELTA;
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
std::tuple<PuckPush, float, PuckPush::Observation, float> PuckPush::Step(
      const PuckPush::Action& action, const PuckPush::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  PuckPush next_sim = *this;
  float reward;

  /* ====== Step 1: Update state. ====== */

  // Sample constants.
  vector_t bot_dir = vector_t(1, 0).rotated(action.orientation +
      std::normal_distribution<float>(0.0f, BOT_MOVE_ORIENTATION_NOISE)(RngDet()));
  float bot_speed;
  do {
    bot_speed = std::normal_distribution<float>(BOT_SPEED, BOT_SPEED_NOISE)(RngDet());
  } while (bot_speed <= 0);
  float puck_bot_roll_constant;
  do {
    puck_bot_roll_constant = std::normal_distribution<float>(PUCK_BOT_ROLL_CONSTANT, PUCK_BOT_ROLL_CONSTANT_NOISE)(RngDet());
  } while (puck_bot_roll_constant <= 0);
  float puck_bot_roll_displacement;
  do {
    puck_bot_roll_displacement = std::normal_distribution<float>(PUCK_BOT_ROLL_DISPLACEMENT, PUCK_BOT_ROLL_DISPLACEMENT_NOISE)(RngDet());
  } while (puck_bot_roll_displacement < 0);

  // Step puck.
  vector_t bot_segment_start = next_sim.bot_position;
  vector_t bot_segment_end = next_sim.bot_position + bot_dir * bot_speed * DELTA;
  float bot_segment_length = (bot_segment_end - bot_segment_start).norm();
  std::optional<float> intersection_ratio = FindFirstRootQuadratic(
      (bot_segment_end - bot_segment_start).squaredNorm(),
      2 * (bot_segment_start - next_sim.puck_position).dot(bot_segment_end - bot_segment_start),
      (bot_segment_start - next_sim.puck_position).squaredNorm() - (PUCK_RADIUS + BOT_RADIUS) * (PUCK_RADIUS + BOT_RADIUS),
      0.0f, 1.0f);
  if (intersection_ratio) {
    vector_t intersecting_bot_pos = bot_segment_start + *intersection_ratio * (bot_segment_end - bot_segment_start);
    float initial_angle = AngleTo(bot_dir, next_sim.puck_position - intersecting_bot_pos);
    float roll_distance = std::min(
        (1 - *intersection_ratio) * bot_segment_length,
        logf(PI / (2 * std::abs(initial_angle))) / puck_bot_roll_constant);
    float final_angle = initial_angle * expf(puck_bot_roll_constant * roll_distance);
    next_sim.puck_position = bot_segment_start +
      std::min(1.0f, *intersection_ratio + roll_distance / bot_segment_length) * (bot_segment_end - bot_segment_start) +
      (1 + 0.001f + puck_bot_roll_displacement) * (PUCK_RADIUS + BOT_RADIUS) * bot_dir.rotated(final_angle);
  }
  // Step bot.
  next_sim.bot_position = bot_segment_end;

  next_sim.step++;

  // Check terminal and calculate rewards.
  if (!boost::geometry::within(next_sim.bot_position, BOARD_REGION_BOT_BOOST)) {
    next_sim._is_terminal = true;
    next_sim._is_failure = true;
    reward = COLLISION_REWARD;
  }
  if (!next_sim._is_terminal) {
    if (!boost::geometry::within(next_sim.puck_position, BOARD_REGION_PUCK_BOOST)) {
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
      reward = COLLISION_REWARD;
    }
  }
  if (!next_sim._is_terminal) {
    if ((next_sim.puck_position - GOAL).norm() <= GOAL_RADIUS - PUCK_RADIUS) {
      next_sim._is_terminal = true;
      reward = GOAL_REWARD;
    }
  }
  if (!next_sim._is_terminal) {
    if (next_sim.step == MAX_STEPS) {
      reward = COLLISION_REWARD;
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }
  if (!next_sim._is_terminal) {
    reward = STEP_REWARD;
  }

  if (reward != STEP_REWARD && reward != GOAL_REWARD && reward != COLLISION_REWARD) {
    throw std::logic_error("INVALID REWARD!");
  }

  /* ====== Step 2: Generate observation. ====== */

  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }
  float log_prob = 0;

  bool in_noisy_region = false;
  for (const auto& noisy_region : NOISY_REGIONS) {
    if (next_sim.puck_position.x >= noisy_region[0].x && next_sim.puck_position.x <= noisy_region[1].x) {
      if (next_sim.puck_position.y >= noisy_region[0].y && next_sim.puck_position.y <= noisy_region[1].y) {
        in_noisy_region = true;
        break;
      }
    }
  }

  if (!observation) {
    new_observation.bot_position = next_sim.bot_position;
    new_observation.bot_position.x += std::normal_distribution<float>(0.0, BOT_OBSERVATION_NOISE)(RngDet());
    new_observation.bot_position.y += std::normal_distribution<float>(0.0, BOT_OBSERVATION_NOISE)(RngDet());
    if (!in_noisy_region) {
      if (std::uniform_real_distribution<float>(0.0f, 1.0f)(RngDet()) < PUCK_FLICKER_PROBABILITY) {
        new_observation.puck_position.x = std::numeric_limits<float>::quiet_NaN();
        new_observation.puck_position.y = std::numeric_limits<float>::quiet_NaN();
      } else {
        new_observation.puck_position = next_sim.puck_position;
        new_observation.puck_position.x += std::normal_distribution<float>(0.0, PUCK_OBSERVATION_NOISE)(RngDet());
        new_observation.puck_position.y += std::normal_distribution<float>(0.0, PUCK_OBSERVATION_NOISE)(RngDet());
      }
    } else {
      new_observation.puck_position.x = std::numeric_limits<float>::quiet_NaN();
      new_observation.puck_position.y = std::numeric_limits<float>::quiet_NaN();
    }
  }
  if (compute_log_prob) {
    log_prob += NormalLogProb(next_sim.bot_position.x, BOT_OBSERVATION_NOISE, new_observation.bot_position.x);
    log_prob += NormalLogProb(next_sim.bot_position.y, BOT_OBSERVATION_NOISE, new_observation.bot_position.y);
    if (!in_noisy_region) {
      if (std::isnan(new_observation.puck_position.x) && std::isnan(new_observation.puck_position.y)) {
        // Flickering outside noisy regions.
        log_prob += logf(PUCK_FLICKER_PROBABILITY);
      } else {
        log_prob += NormalLogProb(next_sim.puck_position.x, PUCK_OBSERVATION_NOISE, new_observation.puck_position.x);
        log_prob += NormalLogProb(next_sim.puck_position.y, PUCK_OBSERVATION_NOISE, new_observation.puck_position.y);
      }
    } else {
      if (std::isnan(new_observation.puck_position.x) && std::isnan(new_observation.puck_position.y)) {
        // Missing observation inside noisy regions.
        log_prob += 0;
      } else {
        log_prob += -std::numeric_limits<float>::infinity();
      }
    }
  }

  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<PuckPush, float, PuckPush::Observation, float> PuckPush::Step<true>(
    const PuckPush::Action& action, const PuckPush::Observation* observation) const;
template std::tuple<PuckPush, float, PuckPush::Observation, float> PuckPush::Step<false>(
    const PuckPush::Action& action, const PuckPush::Observation* observation) const;

/* ====== Serialization functions ====== */

void PuckPush::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  bot_position.Encode(data);
  puck_position.Encode(data);
}

void PuckPush::EncodeContext(list_t<float>& data) {
  if constexpr (RANDOMIZE_GOAL) {
    GOAL.Encode(data);
  }
}

cv::Mat PuckPush::Render(const list_t<PuckPush>& belief_sims,
    const list_t<list_t<Action>>& macro_actions, const vector_t& macro_action_start) const {

  constexpr float SCENARIO_MIN_HORIZONTAL = -30.0f;
  constexpr float SCENARIO_MAX_HORIZONTAL = 1310.0f;
  constexpr float SCENARIO_MIN_VERTICAL = -30.0f;
  constexpr float SCENARIO_MAX_VERTICAL = 617.0f;
  constexpr float RESOLUTION = 1.0f;
  auto to_frame = [&](const vector_t& vector) {
    return cv::Point{
      static_cast<int>((vector.x - SCENARIO_MIN_HORIZONTAL) / RESOLUTION),
      static_cast<int>((vector.y - SCENARIO_MIN_VERTICAL) / RESOLUTION)
    };
  };
  auto to_frame_dist = [&](float d) {
    return static_cast<int>(d / RESOLUTION);
  };

  cv::Mat frame(
      static_cast<int>((SCENARIO_MAX_VERTICAL - SCENARIO_MIN_VERTICAL) / RESOLUTION),
      static_cast<int>((SCENARIO_MAX_HORIZONTAL - SCENARIO_MIN_HORIZONTAL) / RESOLUTION),
      CV_8UC3,
      cv::Scalar(255, 255, 255));

  // Draw noisy
  for (size_t i = 0; i < NOISY_REGIONS.size(); i++) {
    cv::rectangle(frame,
        to_frame(NOISY_REGIONS[i][0] + vector_t{PUCK_RADIUS, PUCK_RADIUS}),
        to_frame(NOISY_REGIONS[i][1] - vector_t{PUCK_RADIUS, PUCK_RADIUS}),
        cv::Scalar(25, 211, 255),
        -1, cv::LINE_AA);
  }

  // Draw regions region.
  cv::rectangle(frame, to_frame(BOARD_REGION[0]), to_frame(BOARD_REGION[2]),
      cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
  cv::rectangle(frame,
      to_frame(BOT_START_REGION[0] - BOT_RADIUS * vector_t{1, 1}),
      to_frame(BOT_START_REGION[2] + BOT_RADIUS * vector_t{1, 1}),
      cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
  cv::rectangle(frame,
      to_frame(PUCK_START_REGION[0] - PUCK_RADIUS * vector_t{1, 1}),
      to_frame(PUCK_START_REGION[2] + PUCK_RADIUS * vector_t{1, 1}),
      cv::Scalar(60, 221, 255), 1, cv::LINE_AA);
  cv::rectangle(frame,
      to_frame(GOAL_REGION[0] - GOAL_RADIUS * vector_t{1, 1}),
      to_frame(GOAL_REGION[2] + GOAL_RADIUS * vector_t{1, 1}),
      cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

  // Draw goal region.
  cv::circle(frame, to_frame(GOAL), to_frame_dist(GOAL_RADIUS),
      cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

  cv::drawMarker(frame, to_frame(BOT_START_POSITION),
      cv::Scalar(255, 255, 0), cv::MARKER_TILTED_CROSS, 20, 4, cv::LINE_AA);
  cv::drawMarker(frame, to_frame(PUCK_START_POSITION),
      cv::Scalar(60, 221, 255), cv::MARKER_TILTED_CROSS, 20, 4, cv::LINE_AA);

  // Draw ego-agent.
  cv::circle(frame, to_frame(bot_position), to_frame_dist(BOT_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(bot_position), to_frame_dist(BOT_RADIUS),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  // Draw puck
  cv::circle(frame, to_frame(puck_position), to_frame_dist(PUCK_RADIUS),
      cv::Scalar(25, 211, 255), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(puck_position), to_frame_dist(PUCK_RADIUS),
      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

  for (auto& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.bot_position), cv::Scalar(0, 0, 255),
        cv::MARKER_CROSS, 2, 1, cv::LINE_4);

    cv::drawMarker(frame, to_frame(belief_sim.puck_position), cv::Scalar(0, 0, 255),
        cv::MARKER_CROSS, 2, 1, cv::LINE_4);
  }

  /*
  vector_t s = macro_action_start;
  for (const Action& a : macro_action) {
    vector_t e = s + vector_t(DELTA * BOT_SPEED, 0).rotated(a.orientation);
    cv::line(frame, to_frame(s), to_frame(e),
        cv::Scalar(75, 156, 0), 2, cv::LINE_AA);
    s = e;
  }
  */

  return frame;
}

}
