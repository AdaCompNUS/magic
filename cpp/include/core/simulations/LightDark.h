#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace simulations {

class LightDark {

public:

  // Environment parameters.
  static constexpr float DELTA = 1.0f;

  static constexpr vector_t EGO_START = vector_t(2.0f, 2.5f);
  static constexpr float EGO_SPAWN_NOISE = 2.0f;
  static constexpr float EGO_RADIUS = 0.5f;
  static constexpr float EGO_SPEED = 0.5f;

  static constexpr float LIGHT_POS = 5.0f;
  static constexpr float LIGHT_MIN_OBSERVATION_NOISE = 0.1f;
  static constexpr float LIGHT_OBSERVATION_NOISE_CONSTANT = 0.5f;

  static constexpr vector_t GOAL = vector_t{0, 0};
  static constexpr float GOAL_ALLOWANCE = DELTA * EGO_SPEED * sqrtf(2);

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 40;
  static constexpr float STEP_REWARD = -0.1f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 40;
  static constexpr size_t PLANNING_TIME = 100;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 50;
  static constexpr float POMCPOW_K_ACTION = 12.5f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.075f;
  static constexpr float POMCPOW_K_OBSERVATION = 5.0f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.05f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 1000;

  struct Observation {
    vector_t ego_agent_position;
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

  /* ====== Construction functions ====== */
  LightDark();
  static LightDark CreateRandom();

  /* ====== Belief related functions ====== */
  static LightDark SampleBeliefPrior();
  float Error(const LightDark& other) const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  template <bool compute_log_prob>
  std::tuple<LightDark, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  cv::Mat Render(const list_t<LightDark>& belief_sims,
      const list_t<Action>& macro_action={},
      const vector_t& macro_action_start={}) const;

private:

  bool _is_terminal;

};

}
