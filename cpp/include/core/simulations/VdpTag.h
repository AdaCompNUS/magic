#pragma once

#include "core/Types.h"
#include <opencv2/core/mat.hpp>
#include <vector>

/*
 * C++ port of https://github.com/zsunberg/VDPTag2.jl/tree/master/src
 */
namespace simulations {
class VdpTag {

public:

  // Environment parameters.
  static constexpr float MU = 2.0f;
  static constexpr float AGENT_SPEED = 1.0f;
  static constexpr float DELTA = 0.5f;
  static constexpr float RK4_STEP_SIZE = 0.1f;
  static constexpr size_t RK4_STEP_ITER = 5;
  static constexpr float TAG_RADIUS = 0.1f;
  static constexpr float POS_STD = 0.05f;
  static constexpr float ACTIVE_MEAS_STD = 0.1f;
  static constexpr float MEAS_STD = 5.0f;

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 10000;
  static constexpr size_t BELIEF_NUM_THREADS = 1;

  // Planning related parameters.
  static constexpr float STEP_REWARD = -1.0f;
  static constexpr float TAG_REWARD = 100.0f;
  static constexpr float ACTIVE_MEAS_REWARD = -5.0f;
  static constexpr float GAMMA = 0.98f;
  static constexpr float WORST_REWARD = (STEP_REWARD + ACTIVE_MEAS_REWARD) / (1.0f - GAMMA);
  static constexpr size_t SEARCH_DEPTH = 10;
  static constexpr size_t PLANNING_TIME = 1000;

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 110;
  static constexpr float POMCPOW_K_ACTION = 30.0f;
  static constexpr float POMCPOW_ALPHA_ACTION = 1.0f / 30;
  static constexpr float POMCPOW_K_OBSERVATION = 5.0f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 1.0f / 100;

  // Experiment related parameters.
  static constexpr size_t MAX_STEPS = 100;

  struct Observation {
    array_t<float, 8> beam_distances;
    uint64_t Discretize() const;
  };

  struct Action {
    bool look;
    float angle;
    static Action Rand();
    Action() {}
    Action(float angle) : look(false), angle(angle) { }
    float Id() const { return look ? 1000.0f + angle : angle; }

    static list_t<list_t<Action>> CreateHandcrafted(size_t length);
  };

  size_t step;
  vector_t ego_agent_position;
  vector_t exo_agent_position;

  /* ====== Construction related functions ====== */
  VdpTag();
  static VdpTag CreateRandom();

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Belief related functions ====== */
  static VdpTag SampleBeliefPrior();
  float Error(const VdpTag& other) const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  bool IsFailure() const { return _is_failure; }
  template <bool compute_log_prob>
  std::tuple<VdpTag, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  /* ====== Serialization functions ====== */
  void Encode(list_t<float>& data) const;
  cv::Mat Render(const list_t<VdpTag>& belief_sims) const;

private:

  bool _is_terminal;
  bool _is_failure;
  static float Cross(const vector_t& a, const vector_t& b);
  vector_t VdpDynamics(const vector_t& v) const;
  vector_t Rk4Step(const vector_t& v) const;
  vector_t BarrierStop(const vector_t& v, const vector_t& d) const;
  size_t ActiveBeam(const vector_t& v) const;

};
}
