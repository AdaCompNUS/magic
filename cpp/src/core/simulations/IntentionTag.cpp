#include "core/simulations/IntentionTag.h"

#include "core/Util.h"
#include "rvo2/RVO.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <stdexcept>

namespace simulations {

IntentionTag::Action IntentionTag::Action::Rand() {
  Action action{std::uniform_real_distribution<float>(0, 2 * PI)(Rng())};
  action.trigger = std::bernoulli_distribution(0.2)(Rng());
  return action;
}

uint64_t IntentionTag::Observation::Discretize() const {
  array_t<int, 2> data;
  data[0] = static_cast<int>(floorf(target_agent_position.x / 0.5f));
  data[1] = static_cast<int>(floorf(target_agent_position.y / 0.5f));
  return boost::hash_value(data);
}

/* ====== Construction functions ====== */

IntentionTag::IntentionTag() : step(0), _is_terminal(false) {

}

IntentionTag IntentionTag::CreateRandom() {

  // Clear adversarial agent initial positions, so that
  // SampledBeliefPrior will select randomly spawned ones.
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    _adversarial_agent_initial_positions[i] = {};
  }

  IntentionTag sim = SampleBeliefPrior();

  // Reset RVOs.
  if (_rvo) {
    delete _rvo;
  }
  _rvo = new RVO::RVOSimulator();

  // Initialize RVO.
  _rvo->setTimeStep(static_cast<float>(DELTA));
  _rvo->processObstacles();

  // Add adversarial agents.
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    _rvo_ids[i] = _rvo->addAgent(
        ToRVO(sim.adversarial_agent_positions[i]), // Position.
        8.0, // Neighbour distance.
        10, // Max neighbours.
        static_cast<float>(DELTA), // Time horizon.
        static_cast<float>(DELTA), // Time horizon (obstacles).
        static_cast<float>(ADVERSARIAL_RADIUS),
        static_cast<float>(ADVERSARIAL_SPEED), // Max speed.
        ToRVO(sim.adversarial_agent_previous_velocities[i]));
    _adversarial_agent_initial_positions[i] = sim.adversarial_agent_positions[i];
  }

  return sim;
}

/* ====== Belief related functions ======*/

IntentionTag IntentionTag::SampleBeliefPrior() {

  auto random_disc_point = [&](float radius) {
    float r = radius * sqrtf(std::uniform_real_distribution<float>(0.0, 1.0)(Rng()));
    float theta = std::uniform_real_distribution<float>(0.0, 2 * PI)(Rng());
    return vector_t(r, 0).rotated(theta);
  };

  IntentionTag sim;
  while (true) {
    sim.ego_agent_position = EGO_AGENT_START;

    sim.target_agent_position = TARGET_AGENT_START;
    sim.target_agent_intention = std::uniform_int_distribution<size_t>(0, NUM_INTENTIONS - 1)(Rng());
    if (sim.target_agent_position.norm() >= RADIUS - TARGET_RADIUS) {
      continue;
    }

    for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
      if (!_adversarial_agent_initial_positions[i]) {
        if (i == 0) {
          sim.adversarial_agent_positions[i] = {0, 0};
        } else {
          sim.adversarial_agent_positions[i] = random_disc_point(RADIUS - ADVERSARIAL_RADIUS);
        }
        sim.adversarial_agent_previous_velocities[i] = {0, 0};
      } else {
        sim.adversarial_agent_positions[i] = *_adversarial_agent_initial_positions[i];
      }
    }

    bool has_violation = false;
    for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
      if ((sim.adversarial_agent_positions[i] - sim.ego_agent_position).norm() <= EGO_RADIUS + ADVERSARIAL_RADIUS + ADVERSARIAL_SPAWN_CLEARANCE) {
        has_violation = true;
        break;
      }
      for (size_t j = i + 1; j < NUM_ADVERSARIAL_AGENTS; j++) {
        if ((sim.adversarial_agent_positions[i] - sim.adversarial_agent_positions[j]).norm() <= 2 * ADVERSARIAL_RADIUS + ADVERSARIAL_SPAWN_CLEARANCE / 2) {
          has_violation = true;
          break;
        }
      }
    }
    if (has_violation) {
      continue;
    }

    break;
  }

  return sim;
}

float IntentionTag::Error(const IntentionTag& other) const {
  float error = (target_agent_position - other.target_agent_position).norm();
  error += (ego_agent_position - other.ego_agent_position).norm();
  return error / 2;
}

/* ====== Bounds related functions ====== */
// Actually a high probability estimate, assuming actuation noise samples below 3 s.d.
float IntentionTag::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.

  float distance = std::max(0.0f, (target_agent_position - ego_agent_position).norm() - (EGO_RADIUS + TARGET_RADIUS));
  float max_distance_per_step = EGO_SPEED * DELTA + TARGET_SPEED * DELTA + 3 * TARGET_ACTUATION_NOISE;
  size_t steps = static_cast<size_t>(round(ceilf(distance / max_distance_per_step)));
  if (steps <= 1) {
    return GOAL_REWARD;
  } else {
    return (1 - powf(GAMMA, static_cast<float>(steps) - 1)) / (1 - static_cast<float>(steps)) * STEP_REWARD +
      powf(GAMMA, static_cast<float>(steps) - 1) * GOAL_REWARD;
  }
}

/* ====== Stepping functions ====== */

vector_t IntentionTag::Goal(size_t intention) const {
  return vector_t(RADIUS, 0).rotated(static_cast<float>(intention) * 2 * PI / static_cast<float>(NUM_INTENTIONS));
}

vector_t IntentionTag::PreferredVelocity(size_t intention, const vector_t& position, float max_speed) const {
  return (Goal(intention) - position).normalized() * max_speed;
}

template <bool compute_log_prob>
std::tuple<IntentionTag, float, IntentionTag::Observation, float> IntentionTag::Step(
    const IntentionTag::Action& action, const IntentionTag::Observation* observation) const {
  if (_is_terminal) { throw std::logic_error("Cannot step terminal simulation."); }

  IntentionTag next_sim = *this;
  float reward;

  /* ====== Step 1: Update state.  ======*/
  next_sim.ego_agent_position += DELTA * vector_t(EGO_SPEED, 0).rotated(action.orientation);
  next_sim.ego_agent_position.x += std::normal_distribution<float>(0.0f, EGO_ACTUATION_NOISE)(RngDet());
  next_sim.ego_agent_position.y += std::normal_distribution<float>(0.0f, EGO_ACTUATION_NOISE)(RngDet());

  // Add adversarial agents.
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    _rvo->setAgentPosition(_rvo_ids[i], ToRVO(adversarial_agent_positions[i]));
    _rvo->setAgentVelocity(_rvo_ids[i], ToRVO(adversarial_agent_previous_velocities[i]));
    _rvo->setAgentPrefVelocity(_rvo_ids[i], ToRVO(
        (ego_agent_position - adversarial_agent_positions[i]).normalized() * ADVERSARIAL_SPEED));
  }
  // Step RVO.
  _rvo->doStep();

  // Update adversarial agents.
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    next_sim.adversarial_agent_positions[i] += DELTA * FromRVO(_rvo->getAgentVelocity(_rvo_ids[i]));
    next_sim.adversarial_agent_previous_velocities[i] = FromRVO(_rvo->getAgentVelocity(_rvo_ids[i]));
  }

  // Update target agent.
  next_sim.target_agent_position += DELTA * PreferredVelocity(target_agent_intention,
        target_agent_position, TARGET_SPEED);
  next_sim.target_agent_position.x += std::normal_distribution<float>(0.0f, TARGET_ACTUATION_NOISE)(RngDet());
  next_sim.target_agent_position.y += std::normal_distribution<float>(0.0f, TARGET_ACTUATION_NOISE)(RngDet());
  next_sim.target_agent_intention = target_agent_intention;
  if ((next_sim.target_agent_position - Goal(next_sim.target_agent_intention)).norm() <= TARGET_RADIUS) {
    next_sim.target_agent_intention = std::uniform_int_distribution<size_t>(0, NUM_INTENTIONS - 1)(RngDet());
  }

  next_sim.step++;

  // Check terminal and calculate rewards.
  if (!next_sim._is_terminal) {
    if (next_sim.ego_agent_position.norm() >= RADIUS - EGO_RADIUS) {
      next_sim._is_terminal = true;
      reward = COLLISION_REWARD;
    }
  }
  if (!next_sim._is_terminal) {
    for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
      if ((next_sim.adversarial_agent_positions[i] - next_sim.ego_agent_position).norm() <= EGO_RADIUS + ADVERSARIAL_RADIUS) {
        next_sim._is_terminal = true;
        reward = COLLISION_REWARD;
        break;
      }
    }
  }
  if (!next_sim._is_terminal && action.trigger) {
    if ((next_sim.target_agent_position - next_sim.ego_agent_position).norm() <= EGO_RADIUS + TARGET_RADIUS) {
      next_sim._is_terminal = true;
      reward = GOAL_REWARD;
    } else {
      next_sim._is_terminal = true;
      reward = COLLISION_REWARD;
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

  float target_observation_noise = TARGET_OBSERVATION_NOISE_CONSTANT * next_sim.ego_agent_position.norm() + TARGET_OBSERVATION_NOISE_BASE;
  if (!observation) {
    new_observation.target_agent_position = next_sim.target_agent_position;
    new_observation.target_agent_position.x += std::normal_distribution<float>(0.0, target_observation_noise)(RngDet());
    new_observation.target_agent_position.y += std::normal_distribution<float>(0.0, target_observation_noise)(RngDet());
  }
  if constexpr (compute_log_prob) {
    log_prob += NormalLogProb(next_sim.target_agent_position.x, target_observation_noise, new_observation.target_agent_position.x);
    log_prob += NormalLogProb(next_sim.target_agent_position.y, target_observation_noise, new_observation.target_agent_position.y);
  }


  return std::make_tuple(next_sim, reward, observation ? Observation() : new_observation, log_prob);
}
template std::tuple<IntentionTag, float, IntentionTag::Observation, float> IntentionTag::Step<true>(
    const IntentionTag::Action& action, const IntentionTag::Observation* observation) const;
template std::tuple<IntentionTag, float, IntentionTag::Observation, float> IntentionTag::Step<false>(
    const IntentionTag::Action& action, const IntentionTag::Observation* observation) const;

/* ====== Serialization functions ====== */

void IntentionTag::Encode(list_t<float>& data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
  target_agent_position.Encode(data);
  data.emplace_back(static_cast<float>(target_agent_intention));
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    adversarial_agent_positions[i].Encode(data);
    adversarial_agent_previous_velocities[i].Encode(data);
  }
}

cv::Mat IntentionTag::Render(const list_t<IntentionTag>& belief_sims) const {

  constexpr float SCENARIO_MIN = -12.0f;
  constexpr float SCENARIO_MAX = 12.0f;
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

  cv::drawMarker(frame, to_frame(EGO_AGENT_START), cv::Scalar(255, 255, 0),
      cv::MARKER_TILTED_CROSS, 20, 3, cv::LINE_AA);

  // Draw belief intention intensities.
  array_t<size_t, NUM_INTENTIONS> intention_intensities;
  intention_intensities.fill(0);
  for (const simulations::IntentionTag& belief_sim : belief_sims) {
    intention_intensities[belief_sim.target_agent_intention]++;
  }
  for (size_t i = 0; i < NUM_INTENTIONS; i++) {
    double v = static_cast<double>(255 - 255 *
        static_cast<float>(intention_intensities[i]) / static_cast<float>(belief_sims.size()));
    cv::circle(frame,
        to_frame(vector_t(RADIUS, 0)
            .rotated(static_cast<float>(i) * 2 * PI / static_cast<float>(NUM_INTENTIONS))),
        to_frame_dist(0.4f),
        cv::Scalar(v, v, v), 5, cv::LINE_AA);
  }

  // Draw boundary.
  cv::circle(frame, to_frame({0, 0}), to_frame_dist(RADIUS),
      cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  // Draw possible intentions.
  for (size_t i = 0; i < NUM_INTENTIONS; i++) {
    cv::circle(frame,
        to_frame(vector_t(RADIUS, 0)
            .rotated(static_cast<float>(i) * 2 * PI / static_cast<float>(NUM_INTENTIONS))),
        to_frame_dist(0.2f),
        cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
  }

  // Draw ego agent.
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(ego_agent_position), to_frame_dist(EGO_RADIUS),
      cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  // Draw adversarial agents.
  for (size_t i = 0; i < NUM_ADVERSARIAL_AGENTS; i++) {
    cv::circle(frame, to_frame(adversarial_agent_positions[i]), to_frame_dist(ADVERSARIAL_RADIUS),
        cv::Scalar(198, 184, 254), -1, cv::LINE_AA);
    cv::circle(frame, to_frame(adversarial_agent_positions[i]), to_frame_dist(ADVERSARIAL_RADIUS),
        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }

  // Draw target agent.
  cv::circle(frame, to_frame(target_agent_position), to_frame_dist(TARGET_RADIUS),
      cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
  cv::circle(frame, to_frame(target_agent_position), to_frame_dist(TARGET_RADIUS),
      cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  cv::Mat beacon_img = cv::imread("/home/leeyiyuan/beacon.png", cv::IMREAD_UNCHANGED);
  cv::resize(beacon_img, beacon_img, cv::Size(30, 30), 0, 0);
  cv::Mat channels[4];
  cv::split(beacon_img, channels);
  list_t<cv::Mat> merge_channels = {channels[0], channels[1], channels[2]};
  cv::merge(merge_channels, beacon_img);
  beacon_img.copyTo(
      frame.rowRange((frame.rows - beacon_img.rows) / 2, (frame.rows + beacon_img.rows) / 2)
      .colRange((frame.cols - beacon_img.cols) / 2, (frame.cols + beacon_img.cols) / 2),
      channels[3]);

  // Draw belief positions.
  for (const simulations::IntentionTag& belief_sim : belief_sims) {
    /*
    cv::drawMarker(frame, to_frame(belief_sim.ego_agent_position), cv::Scalar(0, 0, 255),
        cv::MARKER_CROSS, 2, 1, cv::LINE_4);
    */
    cv::drawMarker(frame, to_frame(belief_sim.target_agent_position), cv::Scalar(0, 211, 255),
        cv::MARKER_CROSS, 1, 1, cv::LINE_4);
  }

  return frame;
}

}
