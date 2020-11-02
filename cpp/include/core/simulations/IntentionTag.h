#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <rvo2/RVOSimulator.h>
#include <vector>

namespace simulations {

class IntentionTag {

public:

  // Environment parameters.
  static constexpr float RADIUS = 10.0f;
  static constexpr float DELTA = 1.0f;

  static constexpr size_t NUM_INTENTIONS = 8;

  static constexpr float EGO_RADIUS = 0.5f;
  static constexpr float EGO_SPEED = 0.5f;
  static constexpr vector_t EGO_AGENT_START = {-8, 0};
  static constexpr float EGO_ACTUATION_NOISE = 0.05f;
  static constexpr float EGO_OBSERVATION_NOISE = 0.1f;

  static constexpr float TARGET_RADIUS = 0.5f;
  static constexpr float TARGET_SPEED = 0.25f;
  static constexpr vector_t TARGET_AGENT_START = {8, 0};
  static constexpr float TARGET_OBSERVATION_NOISE_BASE = 0.1f;
  static constexpr float TARGET_OBSERVATION_NOISE_CONSTANT = 0.5f;
  static constexpr float TARGET_ACTUATION_NOISE = 0.05f;

  static constexpr size_t NUM_ADVERSARIAL_AGENTS = 2;
  static constexpr float ADVERSARIAL_RADIUS = 2.5f;
  static constexpr float ADVERSARIAL_SPEED = 0.1f;
  static constexpr float ADVERSARIAL_SPAWN_CLEARANCE = 3.0f;

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 80;
  static constexpr float STEP_REWARD = -0.1f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 80;
  static constexpr size_t PLANNING_TIME = 100;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 50;
  static constexpr float POMCPOW_K_ACTION = 37.5f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.025f;
  static constexpr float POMCPOW_K_OBSERVATION = 5.0f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.025f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 50;

  struct Observation {
    vector_t ego_agent_position;
    vector_t target_agent_position;
    uint64_t Discretize() const;
  };

  struct Action {
    bool trigger;
    float orientation;
    static Action Rand();
    Action() {}
    Action(float orientation) : trigger(false), orientation(orientation) { }
    float Id() const { return trigger ? std::numeric_limits<float>::quiet_NaN() : orientation; }
  };

  size_t step;
  vector_t ego_agent_position;
  vector_t target_agent_position;
  size_t target_agent_intention;

  array_t<vector_t, NUM_ADVERSARIAL_AGENTS> adversarial_agent_positions;
  array_t<vector_t, NUM_ADVERSARIAL_AGENTS> adversarial_agent_previous_velocities;

  /* ====== Construction functions ====== */
  IntentionTag();
  static IntentionTag CreateRandom();

  /* ====== Belief related functions ====== */
  static IntentionTag SampleBeliefPrior();
  float Error(const IntentionTag& other) const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  template <bool compute_log_prob>
  std::tuple<IntentionTag, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  cv::Mat Render(const list_t<IntentionTag>& belief_sims) const;

  vector_t Goal(size_t intention) const;
  vector_t PreferredVelocity(size_t intention, const vector_t& position, float max_speed) const;

private:

  bool _is_terminal;
  inline static array_t<std::optional<vector_t>, NUM_ADVERSARIAL_AGENTS> _adversarial_agent_initial_positions = {};
  inline static RVO::RVOSimulator* _rvo = NULL;
  inline static array_t<size_t, NUM_ADVERSARIAL_AGENTS> _rvo_ids = {};

};

}