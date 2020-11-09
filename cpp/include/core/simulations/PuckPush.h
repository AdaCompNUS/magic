#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace simulations {

class PuckPush {

public:

  // Environment parameters.
  static constexpr float DELTA = 2.0f;
  static constexpr float BOT_RADIUS = 38;
  static constexpr vector_t BOT_START_POSITION = {300, 299};
  static constexpr vector_t PUCK_START_POSITION = {450, 299};
  static constexpr float INITIAL_POSITION_NOISE = 4;

  static constexpr float BOT_SPEED = 50.0f;
  static constexpr float BOT_SPEED_NOISE = 2.5f;
  static constexpr float BOT_MOVE_ORIENTATION_NOISE = 0.05f;
  static constexpr float BOT_OBSERVATION_NOISE = 3;
  static constexpr float PUCK_RADIUS = 16;
  static constexpr float PUCK_OBSERVATION_NOISE = 5;
  static constexpr float PUCK_FLICKER_PROBABILITY = 0.1f;
  static constexpr float PUCK_BOT_ROLL_CONSTANT = 0.01f;
  static constexpr float PUCK_BOT_ROLL_CONSTANT_NOISE = 0.003f;
  static constexpr float PUCK_BOT_ROLL_DISPLACEMENT = 0.0f;
  static constexpr float PUCK_BOT_ROLL_DISPLACEMENT_NOISE = 0.03f;
  static constexpr vector_t GOAL = {1058, 428};
  static constexpr float GOAL_RADIUS = 80.0f;
  static constexpr array_t<array_t<vector_t, 2>, 2> NOISY_REGIONS{
      // We consider the expanded regions since the puck cannot properly be observed as long
      // as it is being partially occluded by the paper.
      array_t<vector_t, 2>{
        vector_t(590 - PUCK_RADIUS, 40 - PUCK_RADIUS),
        vector_t(700 + PUCK_RADIUS, 558 + PUCK_RADIUS)
      },
      array_t<vector_t, 2>{
        vector_t(790 - PUCK_RADIUS, 40 - PUCK_RADIUS),
        vector_t(900 + PUCK_RADIUS, 558 + PUCK_RADIUS)
      }
  };
  static constexpr array_t<vector_t, 4> BOARD_REGION{
      vector_t{37, 40},
      vector_t{1250, 40},
      vector_t{1250, 558},
      vector_t{37, 558}
  };
  static constexpr array_t<vector_t, 4> BOARD_REGION_BOT{
      BOARD_REGION[0] + BOT_RADIUS * vector_t(1, 1),
      BOARD_REGION[1] + BOT_RADIUS * vector_t(-1, 1),
      BOARD_REGION[2] + BOT_RADIUS * vector_t(-1, -1),
      BOARD_REGION[3] + BOT_RADIUS * vector_t(1, -1)
  };
  static constexpr array_t<vector_t, 4> BOARD_REGION_PUCK{
      BOARD_REGION[0] + PUCK_RADIUS * vector_t(1, 1),
      BOARD_REGION[1] + PUCK_RADIUS * vector_t(-1, 1),
      BOARD_REGION[2] + PUCK_RADIUS * vector_t(-1, -1),
      BOARD_REGION[3] + PUCK_RADIUS * vector_t(1, -1)
  };

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 100;
  static constexpr float GOAL_REWARD = 100;
  static constexpr float STEP_REWARD = -0.1f;
  static constexpr float COLLISION_REWARD = -100;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = COLLISION_REWARD;
  static constexpr size_t SEARCH_DEPTH = 100;
  static constexpr size_t PLANNING_TIME = 100;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 25;
  static constexpr float POMCPOW_K_ACTION = 37.5f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.025f;
  static constexpr float POMCPOW_K_OBSERVATION = 2.5f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.075f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 30;

  struct Observation {
    vector_t bot_position;
    vector_t puck_position;
    uint64_t Discretize() const;
  };

  struct Action {
    float orientation;
    static Action Rand();
    Action() {}
    Action(float orientation) : orientation(orientation) { }
    float Id() const { return orientation; }
  };

  size_t step;
  vector_t bot_position;
  vector_t puck_position;

  /* ====== Construction functions ====== */
  PuckPush();
  static PuckPush CreateRandom();

  /* ====== Belief related functions ====== */
  static PuckPush SampleBeliefPrior();
  float Error(const PuckPush& other) const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  template <bool compute_log_prob>
  std::tuple<PuckPush, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  cv::Mat Render(const list_t<PuckPush>& belief_sims,
      const list_t<Action>& macro_action={},
      const vector_t& macro_action_start={}) const;

private:

  bool _is_terminal;
  bool ValidateKnownConstraints(bool throws=false) const;

};

}
