#pragma once

#include <carla/sumonetwork/SumoNetwork.h>

#include "core/Types.h"
#include "resources/meskel_square_sumo_network.h"
#include "resources/meskel_square_sumo_network_occupancy.h"
#include <opencv2/core/mat.hpp>
#include <carla/sumonetwork/SumoNetwork.h>
#include <carla/segments/SegmentMap.h>
#include <carla/occupancy/OccupancyMap.h>

namespace simulations {

class DriveHard {

public:

  static constexpr float DELTA = 0.2f;
  static constexpr float BELIEF_UPDATE_ALLOWANCE = 0.1f; // Covers updating from last action, and forwarding for planning.

  // Agent physics.
  static constexpr vector_t AGENT_MIN = {-0.53535f, -0.98701f};
  static constexpr vector_t AGENT_MAX = {3.2668f, 0.98701f};
  static constexpr float AGENT_UNSAFE_INFLATION = 1.0f;
  static constexpr float AGENT_WHEEL_BASE = 2.5106f;
  static constexpr float EGO_AGENT_SPEED_MAX = 6.0f;
  static constexpr float EGO_AGENT_ACCEL = 3.0f;
  static constexpr float EXO_AGENT_SPEED_MAX = 4.0f;
  static constexpr float EXO_AGENT_ACCEL = 2.0f;
  static constexpr float AGENT_STEER_MAX = 15.0f * PI / 180.0f;
  static constexpr float AGENT_SPEED_NOISE = 0.1f; // 0.05 for DELTA = 0.1
  static constexpr float AGENT_POSITION_NOISE = 0.1f; // 0.05 for DELTA = 0.1
  static constexpr float AGENT_HEADING_NOISE = 0.02f; // 0.01 for DELTA = 0.1
  static constexpr float EGO_AGENT_PATH_RESOLUTION = 0.1f;
  static constexpr float EXO_AGENT_PATH_RESOLUTION = 1.0f;


  // Environment parameters.
  static constexpr float AGENT_TARGET_LOOKAHEAD = 5.0f;
  static constexpr size_t NUM_EXO_AGENTS = 15;
  static constexpr size_t NUM_LOCAL_AGENTS = 15;
  static constexpr float SPAWN_CLEARANCE = 7;

  // Network-defined geometry.
  inline static const carla::sumonetwork::SumoNetwork SUMO_NETWORK =
      carla::sumonetwork::SumoNetwork::LoadFromXml(resources::MESKEL_SQUARE_SUMO_NETWORK);
  inline static const carla::occupancy::OccupancyMap SUMO_NETWORK_OCCUPANCY =
      carla::occupancy::OccupancyMap::LoadFromWkt(resources::MESKEL_SQUARE_SUMO_NETWORK_OCCUPANCY);
  inline static const carla::segments::SegmentMap SUMO_SEGMENTS = SUMO_NETWORK.CreateSegmentMap();


  // Task defined geometry.
  inline static const carla::occupancy::OccupancyMap JUNCTION_OCCUPANCY = carla::occupancy::OccupancyMap({
    {471.03f, 378.36f}, {483.46f, 410.76f}, {459.30f, 443.16f}, {427.59f, 443.02f},
    {424.94f, 441.77f}, {402.31f, 418.72f}, {426.48f, 376.40f}, {467.54f, 374.73f}
  }).Intersection(SUMO_NETWORK_OCCUPANCY);
  inline static const array_t<carla::occupancy::OccupancyMap, 4> ENTRY_OCCUPANCIES = {
    carla::occupancy::OccupancyMap({{470.33f, 378.38f}, {474.53f, 390.83f}, {482.48f, 390.08f}, {483.38f, 377.78f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{460.58f, 441.70f}, {446.47f, 442.00f}, {446.17f, 454.16f}, {459.68f, 454.61f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{425.51f, 441.43f}, {413.80f, 429.64f}, {402.52f, 440.31f}, {413.54f, 452.62f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{426.29f, 376.91f}, {451.70f, 376.50f}, {453.78f, 361.39f}, {428.80f, 357.66f}}).Intersection(SUMO_NETWORK_OCCUPANCY)
  };
  inline static const array_t<carla::occupancy::OccupancyMap, 4> EXIT_OCCUPANCIES = {
    carla::occupancy::OccupancyMap({{478.26f, 398.58f}, {487.70f, 397.91f}, {556.91f, 402.38f}, {555.77f, 415.20f}, {483.01f, 411.01f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{440.81f, 442.21f}, {432.38f, 552.29f}, {419.03f, 551.95f}, {427.31f, 442.10f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{402.50f, 417.68f}, {414.85f, 430.84f}, {340.70f, 500.13f}, {327.72f, 485.45f}}).Intersection(SUMO_NETWORK_OCCUPANCY),
    carla::occupancy::OccupancyMap({{467.79f, 375.95f}, {450.02f, 376.41f}, {468.49f, 248.63f}, {486.84f, 253.13f}}).Intersection(SUMO_NETWORK_OCCUPANCY)
  };
  static constexpr vector_t EXO_SPAWN_BOUNDS_MIN = {400.0f, 350.0f};
  static constexpr vector_t EXO_SPAWN_BOUNDS_MAX = {500.0f, 450.0f};
  inline static carla::segments::SegmentMap EXO_SPAWN_SEGMENTS = SUMO_SEGMENTS.Intersection(
      carla::occupancy::OccupancyMap(EXO_SPAWN_BOUNDS_MIN, EXO_SPAWN_BOUNDS_MAX));

  // Randomization over initialization and context.
  inline static array_t<carla::sumonetwork::RoutePoint, 1500> EGO_PATH;
  inline static carla::occupancy::OccupancyMap EGO_SUMO_NETWORK_OCCUPANCY;

  // Belief tracking related parameters.
  static constexpr size_t BELIEF_NUM_PARTICLES = 500;
  static constexpr size_t BELIEF_NUM_THREADS = 1;

  // Planning related parameters.
  static constexpr size_t MAX_STEPS = 150;
  static constexpr size_t SEARCH_DEPTH = 40;
  inline static size_t PLANNING_TIME;
  static constexpr float GAMMA = 0.98f;
  static constexpr float COLLISION_REWARD = -100.0f;
  static constexpr float PROGRESS_REWARD_WEIGHT = 1.0f;
  static constexpr float LOW_SPEED_THRESHOLD = EGO_AGENT_SPEED_MAX / 2.0f;
  static constexpr float LOW_SPEED_PENALTY = -1.0f * COLLISION_REWARD * (1 - GAMMA) / (1 - powf(GAMMA, SEARCH_DEPTH));

  // POMCPOW related parameters.
  static constexpr float POMCPOW_UCB = 25;
  static constexpr float POMCPOW_K_ACTION = 12.5f;
  static constexpr float POMCPOW_ALPHA_ACTION = 0.025f;
  static constexpr float POMCPOW_K_OBSERVATION = 7.5f;
  static constexpr float POMCPOW_ALPHA_OBSERVATION = 0.05f;

  // DESPOT related parameters.
  static constexpr size_t DESPOT_NUM_SCENARIOS = 50;

  // Serialization settings.
  static constexpr size_t CONTEXT_PATH_INTERVAL = 10;

  struct Agent {
    vector_t position;
    vector_t heading;
    float speed = 0.0f;
    float steer = 0.0f;
  };

  struct Observation {
    Agent ego_agent;
    list_t<Agent> exo_agents;
    uint64_t Discretize() const;
  };

  // Additional fields part of state, but are deterministic functions of (s, a, o).
  struct ObservableState : Observation {
    size_t ego_agent_progress_index;
    size_t step;
    float distance;

    inline static ObservableState From(const Observation& obs) {
      ObservableState sim;
      sim.ego_agent = obs.ego_agent;
      sim.exo_agents = obs.exo_agents;
      sim.ego_agent_progress_index = 0;
      sim.step = 0;
      sim.distance = 0;
      return sim;
    }
  };

  struct UnobservableState {
    list_t<carla::sumonetwork::RoutePoint> exo_agent_intentions;
  };

  struct Action {
    float speed;
    float steer;
    static Action Rand();
    Action() {}
    Action(float speed, float steer): speed(speed), steer(steer) { }
    float Id() const { return 10000 * speed + steer; }

    static list_t<list_t<Action>> CreateHandcrafted(size_t length);
    static list_t<list_t<Action>> Deserialize(const list_t<float>& params, size_t length);
  };

  ObservableState observable_state;
  UnobservableState unobservable_state;
  size_t planning_depth_limit;

  /* ====== Construction functions ====== */
  DriveHard();
  static DriveHard CreateRandom(const list_t<vector_t>& given_path={});
  DriveHard CreateLocal() const;

  /* ====== Bounds related functions ====== */
  float BestReward() const;

  /* ====== Stepping functions ====== */
  bool IsTerminal() const { return _is_terminal; }
  bool IsFailure() const { return _is_failure; }
  template <bool compute_log_prob>
  std::tuple<DriveHard, float, Observation, float> Step(
      const Action& action, const Observation* observation=nullptr) const;

  // Serialization functions.
  void Encode(list_t<float>& data) const;
  void EncodeContext(list_t<float>& data);
  cv::Mat Render(const list_t<DriveHard>& belief_sims,
      const list_t<Action>& macro_action={},
      const vector_t& macro_action_start={}) const;

private:

  bool _is_terminal;
  bool _is_failure;
  static array_t<vector_t, 4> GetBoundingBoxCorners(const Agent& agent, float inflate=0.0f);
  static array_t<vector_t, 4> GetGammaBoundingBoxCorners(const Agent& agent);
  static Agent Actuate(const Agent& agent, float speed, float accel, float steer);
  static float ActuateLogProb(const Agent& agent, float speed, float accel, float steer, const Agent& result);


  vector_t PositionToEgoCentric(const vector_t& pos) const;
  vector_t HeadingToEgoCentric(const vector_t& pos) const;
  void EncodeNonEgoCentric(const Agent& agent, list_t<float>& data) const;
  void EncodeEgoCentric(const Agent& agent, list_t<float>& data) const;
  void EncodeEgoCentric(const carla::sumonetwork::RoutePoint& rp, list_t<float>& data) const;

  static std::tuple<carla::sumonetwork::RoutePoint, size_t, float> StepRoutePoint(
      const vector_t& position, const carla::sumonetwork::RoutePoint& rp, float distance);
  static std::tuple<size_t, size_t> StepEgoRoutePoint(const vector_t& position, size_t index);
};

}

