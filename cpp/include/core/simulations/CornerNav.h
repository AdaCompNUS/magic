#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>
#include <rvo2/RVOSimulator.h>

namespace simulations {

class CornerNav {

public:

  // Environment parameters.
  static constexpr float DELTA = 1.0f;
  static constexpr float EGO_SPAWN_CLEARANCE = 3.0f;

  static constexpr float EGO_RADIUS = 0.5f;
  static constexpr float EGO_SPEED = 1.0f;
  static constexpr vector_t EGO_AGENT_START = {-10.5, 13.75};
  static constexpr float EGO_SPAWN_NOISE = 0.3f;
  static constexpr float EGO_ACTUATION_NOISE = 0.1f;
  static constexpr float EGO_OBSERVATION_NOISE = 0.3f;

  static constexpr vector_t GOAL = {-10.5, -13.75};
  static constexpr float GOAL_RADIUS = 1.0f;

  static constexpr array_t<vector_t, 8> REGION = {
      vector_t{-12.5f, 20},
      vector_t{12.5f, 20},
      vector_t{12.5f, -20},
      vector_t{-12.5f, -20},
      vector_t{-12.5f, -7.5f},
      vector_t{0, -7.5f},
      vector_t{0, 7.5f},
      vector_t{-12.5f, 7.5f}
  };

  static constexpr size_t NUM_EXO_AGENTS = 3;
  static constexpr float EXO_RADIUS = 2.5f;
  static constexpr float EXO_SPEED = 0.2f;
  static constexpr array_t<vector_t, NUM_EXO_AGENTS> EXO_AGENT_START = {
      vector_t{6.25f, 13.75f},
      vector_t{6.25f, 0},
      vector_t{6.25f, -13.75f},
  };
  static constexpr float EXO_SPAWN_NOISE = 0.3f;
  static constexpr float EXO_ACTUATION_NOISE = 0.1f;
  static constexpr float EXO_OBSERVATION_NOISE = 0.3f;

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 100;
  static constexpr float STEP_REWARD = -1;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 100;
  static constexpr size_t PLANNING_TIME = 150;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 25;
  static constexpr float POMCPOW_K_ACTION = 50.0f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.1f;
  static constexpr float POMCPOW_K_OBSERVATION = 10.0f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.075f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 50;

  struct Observation {
    vector_t ego_agent_position;
    array_t<vector_t, NUM_EXO_AGENTS> exo_agent_positions;
    uint64_t Discretize() const;
  };

  struct Action {
    float orientation;
    static Action Rand();
    float Id() const { return orientation; }
  };

  size_t step;
  vector_t ego_agent_position;
  array_t<vector_t, NUM_EXO_AGENTS> exo_agent_positions;
  array_t<vector_t, NUM_EXO_AGENTS> exo_agent_target_velocities;
  array_t<vector_t, NUM_EXO_AGENTS> exo_agent_previous_velocities;

  /* ====== Construction functions ====== */
  CornerNav();
  static CornerNav CreateRandom();

  /* ====== Belief related functions ====== */
  static CornerNav SampleBeliefPrior();
  float Error(const CornerNav& other) const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  template <bool compute_log_prob>
  std::tuple<CornerNav, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  cv::Mat Render(const list_t<CornerNav>& belief_sims) const;

private:

  bool _is_terminal;
  inline static RVO::RVOSimulator* _rvo = NULL;
  inline static array_t<size_t, NUM_EXO_AGENTS> _rvo_ids = {};

};

}
