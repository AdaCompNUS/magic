#include "core/simulations/DriveHard.h"

#include "core/Util.h"
#include "carla/gamma/RVOSimulator.h"
#include "carla/gamma/AgentParams.h"
#include "core/RelaxedDubins.h"
#include <boost/functional/hash.hpp>
#include <carla/geom/Vector2D.h>
#include <carla/occupancy/OccupancyMap.h>
#include <carla/segments/SegmentMap.h>
#include <carla/sumonetwork/SumoNetwork.h>
#include <iostream>
#include <macaron/Base64.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

namespace simulations {

/* ====== Action related functions ====== */
DriveHard::Action DriveHard::Action::Rand() {
  float speed = std::uniform_real_distribution<float>(0.0f, EGO_AGENT_SPEED_MAX)(Rng());
  float steer = AGENT_STEER_MAX * std::uniform_real_distribution<float>(-1, 1)(Rng());
  return Action(speed, steer);
}

list_t<list_t<DriveHard::Action>> DriveHard::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions(7);
  macro_actions[0].emplace_back(EGO_AGENT_SPEED_MAX, -AGENT_STEER_MAX);
  macro_actions[1].emplace_back(EGO_AGENT_SPEED_MAX, 0);
  macro_actions[2].emplace_back(EGO_AGENT_SPEED_MAX, AGENT_STEER_MAX);
  macro_actions[3].emplace_back(EGO_AGENT_SPEED_MAX / 2, -AGENT_STEER_MAX / 2);
  macro_actions[4].emplace_back(EGO_AGENT_SPEED_MAX / 2, 0);
  macro_actions[5].emplace_back(EGO_AGENT_SPEED_MAX / 2, AGENT_STEER_MAX / 2);
  macro_actions[6].emplace_back(0, 0);

  for (size_t i = 0; i < macro_actions.size(); i++) {
    for (size_t j = 0; j < (length - 1) / 2; j++) {
      macro_actions[i].emplace_back(macro_actions[i][0]);
    }
    for (size_t j = (length - 1) / 2; j < length - 1; j++) {
      macro_actions[i].emplace_back(macro_actions[i][0].speed, 0);
    }
  }

  return macro_actions;
}

list_t<list_t<DriveHard::Action>> DriveHard::Action::Deserialize(const list_t<float>& params, size_t length) {

  // None-fine action parameterization.
  size_t count = 7;
  if (params.size() != 2 * count) {
    throw std::logic_error("Incorrect params size! Expected" + std::to_string(2 * count) + " but got " + std::to_string(params.size()));
  }
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 2 * count; i += 2) {
    macro_actions.emplace_back();
    float speed = EGO_AGENT_SPEED_MAX * (params[i] + 1) / 2.0f;
    float steer = AGENT_STEER_MAX * params[i + 1];
    macro_actions.back().emplace_back(speed, steer);
    for (size_t j = 0; j < (length - 1) / 2; j++) {
      macro_actions.back().emplace_back(speed, steer);
    }
    for (size_t j = (length - 1) / 2; j < length - 1; j++) {
      macro_actions.back().emplace_back(speed, 0);
    }
  }

  return macro_actions;
}

uint64_t DriveHard::Observation::Discretize() const {
  list_t<int> data;
  data.emplace_back(static_cast<int>(floorf(ego_agent.position.x / 1.0f)));
  data.emplace_back(static_cast<int>(floorf(ego_agent.position.y / 1.0f)));
  data.emplace_back(static_cast<int>(floorf(ego_agent.heading.x / 1.0f)));
  data.emplace_back(static_cast<int>(floorf(ego_agent.heading.y / 1.0f)));
  data.emplace_back(static_cast<int>(floorf(ego_agent.speed / 0.2f)));
  data.emplace_back(static_cast<int>(floorf(ego_agent.steer / 0.2f)));
  for (size_t i = 0; i < exo_agents.size(); i++) {
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].position.x / 1.0f)));
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].position.y / 1.0f)));
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].heading.x / 1.0f)));
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].heading.y / 1.0f)));
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].speed / 0.2f)));
    data.emplace_back(static_cast<int>(floorf(exo_agents[i].steer / 0.2f)));
  }
  return boost::hash_value(data);
}

/* ====== Construction functions ====== */

DriveHard::DriveHard() : _is_terminal(false), _is_failure(false) {
  observable_state.step = 0;
  observable_state.distance = 0;
  planning_depth_limit = MAX_STEPS;
}

DriveHard DriveHard::CreateRandom(const list_t<vector_t>& given_path) {
  carla::occupancy::OccupancyMap aabb_occupancy;
  vector_t ego_start;

  DriveHard sim;
  { // To limit temp variables' scope.

    if (given_path.size() == 0) {
      carla::occupancy::OccupancyMap entry_occupancy = ENTRY_OCCUPANCIES[std::uniform_int_distribution<size_t>(0, ENTRY_OCCUPANCIES.size() - 1)(Rng())];
      carla::segments::SegmentMap entry_segments = SUMO_SEGMENTS.Difference(JUNCTION_OCCUPANCY).Intersection(entry_occupancy);
      ego_start = entry_segments.RandPoint();

      /*
      ego_start = carla::gamma::Vector2(409.0f, 437.0f);
      carla::occupancy::OccupancyMap entry_occupancy;
      for (size_t i = 0; i < ENTRY_OCCUPANCIES.size(); i++) {
        if (ENTRY_OCCUPANCIES[i].Contains(ego_start)) {
          entry_occupancy = ENTRY_OCCUPANCIES[i];
          break;
        }
      }
      */

      // Get RP and set sim value.
      auto rp = SUMO_NETWORK.GetNearestRoutePoint(ego_start);
      sim.observable_state.ego_agent_progress_index = 0;

      // Randomize path, tracking exit at the same time.
      EGO_PATH[0] = rp;
      const carla::occupancy::OccupancyMap* exit_occupancy = nullptr;
      for (size_t i = 1; i < EGO_PATH.size(); i++) {
        list_t<carla::sumonetwork::RoutePoint> next_rps = SUMO_NETWORK.GetNextRoutePoints(rp, EGO_AGENT_PATH_RESOLUTION);
        rp = next_rps[std::uniform_int_distribution<size_t>(0, next_rps.size() - 1)(Rng())];
        EGO_PATH[i] = rp;
        if (exit_occupancy == nullptr) {
          for (const carla::occupancy::OccupancyMap& occupancy : EXIT_OCCUPANCIES) {
            if (occupancy.Contains(SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[i]))) {
              exit_occupancy = &occupancy;
            }
          }
        }
      }

      // Compute overall occupancy for this randomized initialization/context.
      EGO_SUMO_NETWORK_OCCUPANCY = JUNCTION_OCCUPANCY.Union(entry_occupancy).Union(*exit_occupancy);
    } else {

      ego_start = given_path[0];

      // Invert path in a lame way.
      while (true) {
        bool match = true;
        EGO_PATH[0] = SUMO_NETWORK.GetNearestRoutePoint(given_path[0]);
        for (size_t i = 1; i < EGO_PATH.size(); i++) {
          list_t<carla::sumonetwork::RoutePoint> next_rps = SUMO_NETWORK.GetNextRoutePoints(EGO_PATH[i - 1], EGO_AGENT_PATH_RESOLUTION);
          EGO_PATH[i] = next_rps[std::uniform_int_distribution<size_t>(0, next_rps.size() - 1)(Rng())];
          if ((SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[i]) - given_path[i]).Length() > 1.0f) {
            match = false;
            break;
          }
        }
        if (match) {
          break;
        }
      }

      // Compute overall occupancy.
      const carla::occupancy::OccupancyMap* entry_occupancy = nullptr;
      const carla::occupancy::OccupancyMap* exit_occupancy = nullptr;
      for (size_t i = 0; i < given_path.size(); i++) {
        if (entry_occupancy == nullptr) {
          for (const carla::occupancy::OccupancyMap& occupancy : ENTRY_OCCUPANCIES) {
            if (occupancy.Contains(given_path[i])) {
              entry_occupancy = &occupancy;
              break;
            }
          }
        }
        if (exit_occupancy == nullptr) {
          for (const carla::occupancy::OccupancyMap& occupancy : EXIT_OCCUPANCIES) {
            if (occupancy.Contains(given_path[i])) {
              exit_occupancy = &occupancy;
              break;
            }
          }
        }
      }
      EGO_SUMO_NETWORK_OCCUPANCY = JUNCTION_OCCUPANCY.Union(*entry_occupancy).Union(*exit_occupancy);
    }



    // Set remaining state variables.
    sim.observable_state.ego_agent.position = SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[0]);
    sim.observable_state.ego_agent.heading = vector_t((SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[1]) - SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[0]))).normalized();

    // Init AABB.
    aabb_occupancy = aabb_occupancy.Union(carla::occupancy::OccupancyMap(
        ego_start + SPAWN_CLEARANCE * vector_t(-1, -1),
        ego_start + SPAWN_CLEARANCE * vector_t(1, 1)
    ));
  }

  for (size_t i = 0; i < NUM_EXO_AGENTS; i++) {
    while (true) {
      auto rp = SUMO_NETWORK.GetNearestRoutePoint(EXO_SPAWN_SEGMENTS.Difference(aabb_occupancy).RandPoint());
      vector_t rp_pos = SUMO_NETWORK.GetRoutePointPosition(rp);
      auto next_rp = SUMO_NETWORK.GetNextRoutePoints(rp, EXO_AGENT_PATH_RESOLUTION)[0];
      vector_t next_rp_pos = SUMO_NETWORK.GetRoutePointPosition(next_rp);

      vector_t heading = (next_rp_pos - rp_pos).normalized();
      vector_t heading_to_center = ((EXO_SPAWN_BOUNDS_MAX + EXO_SPAWN_BOUNDS_MIN) / 2 - rp_pos).normalized();
      if (heading.dot(heading_to_center) < 0) {
        continue;
      }

      aabb_occupancy = aabb_occupancy.Union(carla::occupancy::OccupancyMap(
          rp_pos + SPAWN_CLEARANCE * vector_t(-1, -1),
          rp_pos + SPAWN_CLEARANCE * vector_t(1, 1)
      ));

      Agent exo_agent;
      exo_agent.position = rp_pos;
      exo_agent.heading = (next_rp_pos - rp_pos).normalized();
      sim.observable_state.exo_agents.emplace_back(exo_agent);
      sim.unobservable_state.exo_agent_intentions.emplace_back(rp);
      break;
    }
  }

  return sim;
}

DriveHard DriveHard::CreateLocal() const {
  DriveHard sim;
  sim.observable_state.ego_agent = observable_state.ego_agent;
  sim.observable_state.ego_agent_progress_index = observable_state.ego_agent_progress_index;
  sim.observable_state.step = observable_state.step;
  sim.observable_state.distance = observable_state.distance;

  list_t<size_t> nearest_exo_agents;
  for (size_t i = 0; i < observable_state.exo_agents.size(); i++) {
    nearest_exo_agents.emplace_back(i);
  }
  std::sort(nearest_exo_agents.begin(), nearest_exo_agents.end(),
      [&](size_t i, size_t j) {
          return (observable_state.exo_agents[i].position - observable_state.ego_agent.position).norm() <
              (observable_state.exo_agents[j].position - observable_state.ego_agent.position).norm();
      });

  for (size_t i = 0; i < NUM_LOCAL_AGENTS; i++) {
    sim.observable_state.exo_agents.emplace_back(observable_state.exo_agents[nearest_exo_agents[i]]);
    sim.unobservable_state.exo_agent_intentions.emplace_back(unobservable_state.exo_agent_intentions[nearest_exo_agents[i]]);
  }

  sim.planning_depth_limit = std::min(observable_state.step + SEARCH_DEPTH, MAX_STEPS);
  return sim;
}


/* ====== Bounds related functions ====== */

float DriveHard::BestReward() const {
  if (_is_terminal) { return 0; } // Return value of 0 needed for DESPOT.
  float max_distance_per_step = 2.0f * EGO_AGENT_SPEED_MAX * DELTA; // Assume curve distance is at most twice of secant.
  size_t steps = planning_depth_limit - observable_state.step;
  return (1 - powf(GAMMA, static_cast<float>(steps))) / (1 - GAMMA) * (PROGRESS_REWARD_WEIGHT * max_distance_per_step);
}

/* ====== Stepping functions ====== */

DriveHard::Agent DriveHard::Actuate(const Agent& agent, float speed, float accel, float steer) {
  if (agent.speed > speed) {
    accel = -accel;
  }

  Agent next_agent = agent;
  float accel_time = std::min(DELTA, (speed - agent.speed) / accel);
  float travelled_distance = // v0t0 + 0.5at0^2 + v1t1
      agent.speed * accel_time + 0.5f * accel * accel_time * accel_time
      + speed * (DELTA - accel_time);

  if (std::abs(steer) < 0.0001f) {
    next_agent.position = agent.position + agent.heading * travelled_distance;
    next_agent.heading = agent.heading;
  } else {
    float radius = AGENT_WHEEL_BASE  / atanf(steer);
    vector_t center = agent.position + radius * agent.heading.rotated(PI / 2);
    float angle = travelled_distance / radius;

    next_agent.position = center + (agent.position - center).rotated(angle);
    next_agent.heading = agent.heading.rotated(angle).normalized();
  }

  next_agent.speed = agent.speed + accel * accel_time +
      std::normal_distribution<float>(0.0f, AGENT_SPEED_NOISE)(RngDet());
  next_agent.steer = steer;
  next_agent.position.x += std::normal_distribution<float>(0.0f, AGENT_POSITION_NOISE)(RngDet());
  next_agent.position.y += std::normal_distribution<float>(0.0f, AGENT_POSITION_NOISE)(RngDet());
  next_agent.heading.rotate(std::normal_distribution<float>(0.0f, AGENT_HEADING_NOISE)(RngDet()));

  return next_agent;
}

float DriveHard::ActuateLogProb(const Agent& agent, float speed, float accel, float steer,
    const Agent& result) {
  if (agent.speed > speed) {
    accel = -accel;
  }

  vector_t next_position;
  vector_t next_heading;

  float accel_time = std::min(DELTA, (speed - agent.speed) / accel);
  float travelled_distance = // v0t0 + 0.5at0^2 + v1t1
      agent.speed * accel_time + 0.5f * accel * accel_time * accel_time
      + speed * (DELTA - accel_time);

  if (std::abs(steer) < 0.0001f) {
    next_position = agent.position + agent.heading * travelled_distance;
    next_heading = agent.heading;
  } else {
    float radius = AGENT_WHEEL_BASE  / atanf(steer);
    vector_t center = agent.position + radius * agent.heading.rotated(PI / 2);
    float angle = travelled_distance / radius;

    next_position = center + (agent.position - center).rotated(angle);
    next_heading = agent.heading.rotated(angle).normalized();
  }

  float log_prob = 0;
  log_prob += NormalLogProb(agent.speed + accel * accel_time, AGENT_SPEED_NOISE, result.speed);
  log_prob += NormalLogProb(next_position.x, AGENT_POSITION_NOISE, result.position.x);
  log_prob += NormalLogProb(next_position.y, AGENT_POSITION_NOISE, result.position.y);
  log_prob += NormalLogProb(0.0f, AGENT_HEADING_NOISE, AngleTo(next_heading, result.heading));

  return log_prob;
}

array_t<vector_t, 4> DriveHard::GetBoundingBoxCorners(const Agent& agent, float inflate) {
  vector_t forward = agent.heading;
  vector_t sideward = agent.heading.rotated(PI / 2);
  array_t<vector_t, 4> corners {
      agent.position + (AGENT_MIN.x - inflate) * forward + (AGENT_MIN.y - inflate) * sideward,
      agent.position + (AGENT_MIN.x - inflate) * forward + (AGENT_MAX.y + inflate) * sideward,
      agent.position + (AGENT_MAX.x + inflate) * forward + (AGENT_MAX.y + inflate) * sideward,
      agent.position + (AGENT_MAX.x + inflate) * forward + (AGENT_MIN.y - inflate) * sideward
  };
  return corners;
}

array_t<vector_t, 4> DriveHard::GetGammaBoundingBoxCorners(const Agent& agent) {
  vector_t forward = agent.heading;
  vector_t sideward = agent.heading.rotated(PI / 2);
  array_t<vector_t, 4> corners {
      agent.position + (AGENT_MIN.x - 0.1f) * forward + (AGENT_MIN.y - 0.5f) * sideward,
      agent.position + (AGENT_MIN.x - 0.1f) * forward + (AGENT_MAX.y + 0.5f) * sideward,
      agent.position + (AGENT_MAX.x + 2.0f) * forward + (AGENT_MAX.y + 0.5f) * sideward,
      agent.position + (AGENT_MAX.x + 2.0f) * forward + (AGENT_MIN.y - 0.5f) * sideward
  };
  return corners;
}

// Computes posterior over unobservable state, and then samples joint state post-step.
std::tuple<carla::sumonetwork::RoutePoint, size_t, float> DriveHard::StepRoutePoint(
    const vector_t& position, const carla::sumonetwork::RoutePoint& rp, float distance) {

  carla::sumonetwork::RoutePoint result_rp = rp;
  vector_t result_rp_pos = SUMO_NETWORK.GetRoutePointPosition(rp);
  size_t steps = 0;
  float log_prob = 0;

  while (true) {
    list_t<carla::sumonetwork::RoutePoint> next_rps = SUMO_NETWORK.GetNextRoutePoints(result_rp, distance);
    carla::sumonetwork::RoutePoint next_rp = next_rps[std::uniform_int_distribution<size_t>(0, next_rps.size() - 1)(RngDet())];
    vector_t next_rp_pos = SUMO_NETWORK.GetRoutePointPosition(next_rp);
    if ((position - result_rp_pos).norm() < (position - next_rp_pos).norm()) {
      break;
    }
    result_rp = next_rp;
    result_rp_pos = next_rp_pos;
    steps++;
    log_prob += log(static_cast<float>(next_rps.size()));
  }

  return std::make_tuple(result_rp, steps, log_prob);
}

std::tuple<size_t, size_t> DriveHard::StepEgoRoutePoint(
    const vector_t& position, size_t index) {

  size_t result_index = index;
  vector_t result_rp_pos = SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[index]);
  size_t steps = 0;

  while (result_index < EGO_PATH.size() - 2) { // Final index can at most be one before the last element. Not last, since we need the last to compute heading.
    carla::sumonetwork::RoutePoint next_rp = EGO_PATH[result_index + 1];
    vector_t next_rp_pos = SUMO_NETWORK.GetRoutePointPosition(next_rp);
    if ((position - result_rp_pos).norm() < (position - next_rp_pos).norm()) {
      break;
    }
    result_index++;
    result_rp_pos = next_rp_pos;
    steps++;
  }

  return std::make_tuple(result_index, steps);
}

template <bool compute_log_prob>
std::tuple<DriveHard, float, DriveHard::Observation, float> DriveHard::Step(
    const DriveHard::Action& action, const DriveHard::Observation* observation) const {

  DriveHard next_sim = *this;
  float reward = std::numeric_limits<float>::signaling_NaN();
  float log_prob = 0;

  /* ====== Step 1: Update state.  ======*/
  carla::gamma::RVOSimulator gamma;

  // Add ego-agent.
  {
    gamma.addAgent(carla::gamma::AgentParams::getDefaultAgentParam("Car"), 0);
    gamma.setAgentPosition(0, observable_state.ego_agent.position);
    gamma.setAgentVelocity(0, observable_state.ego_agent.speed * observable_state.ego_agent.heading);
    gamma.setAgentHeading(0, observable_state.ego_agent.heading);
    array_t<vector_t, 4> corners = GetGammaBoundingBoxCorners(observable_state.ego_agent);
    gamma.setAgentBoundingBoxCorners(0, list_t<carla::gamma::Vector2>(corners.begin(), corners.end()));
    // This is not necessary since we do not use GAMMA to actuate the ego-agent.
    gamma.setAgentPrefVelocity(0, observable_state.ego_agent.speed * observable_state.ego_agent.heading);
  }
  // Add exo-agents.
  size_t running_gamma_id = 1;
  for (size_t i = 0; i < observable_state.exo_agents.size(); i++) {
    gamma.addAgent(carla::gamma::AgentParams::getDefaultAgentParam("Car"), static_cast<int>(running_gamma_id));
    gamma.setAgentPosition(running_gamma_id, observable_state.exo_agents[i].position);
    gamma.setAgentVelocity(running_gamma_id, observable_state.exo_agents[i].speed * observable_state.exo_agents[i].heading);
    gamma.setAgentHeading(static_cast<int>(running_gamma_id), observable_state.exo_agents[i].heading);
    array_t<vector_t, 4> corners = GetGammaBoundingBoxCorners(observable_state.exo_agents[i]);
    gamma.setAgentBoundingBoxCorners(static_cast<int>(running_gamma_id), list_t<carla::gamma::Vector2>(corners.begin(), corners.end()));
    vector_t rp_pos = SUMO_NETWORK.GetRoutePointPosition(
        unobservable_state.exo_agent_intentions[i]);
    vector_t next_rp_pos = SUMO_NETWORK.GetRoutePointPosition(SUMO_NETWORK.GetNextRoutePoints(
        unobservable_state.exo_agent_intentions[i], EXO_AGENT_PATH_RESOLUTION)[0]);
    vector_t target_rp_pos = SUMO_NETWORK.GetRoutePointPosition(
        SUMO_NETWORK.GetNextRoutePoints(unobservable_state.exo_agent_intentions[i], AGENT_TARGET_LOOKAHEAD)[0]);
    gamma.setAgentPathForward(running_gamma_id, (next_rp_pos - rp_pos).normalized());
    gamma.setAgentPrefVelocity(running_gamma_id, (target_rp_pos - observable_state.exo_agents[i].position).normalized() * EXO_AGENT_SPEED_MAX);
    gamma.setAgentBehaviorType(static_cast<int>(running_gamma_id), carla::gamma::AgentBehaviorType::Gamma);
    running_gamma_id++;
  }
  gamma.doStep();

  // Actuate ego-agent, overriding whith observation if necessary.
  if (observation == nullptr) {
    next_sim.observable_state.ego_agent = Actuate(next_sim.observable_state.ego_agent, action.speed, EGO_AGENT_ACCEL, action.steer);
  } else {
    next_sim.observable_state.ego_agent = observation->ego_agent;
  }
  log_prob += ActuateLogProb(observable_state.ego_agent, action.speed, EGO_AGENT_ACCEL, action.steer, next_sim.observable_state.ego_agent);

  // Step progress.
  next_sim.observable_state.step++;

  // Step progress.
  std::tuple<size_t, size_t> ego_agent_rp_step = StepEgoRoutePoint(next_sim.observable_state.ego_agent.position, observable_state.ego_agent_progress_index);
  next_sim.observable_state.ego_agent_progress_index = std::get<0>(ego_agent_rp_step);
  next_sim.observable_state.distance += static_cast<float>(std::get<1>(ego_agent_rp_step)) * EGO_AGENT_PATH_RESOLUTION;

  // Cache ego-agent bb.
  array_t<vector_t, 4> ego_agent_bb_corners = GetBoundingBoxCorners(next_sim.observable_state.ego_agent);

  // Actuate exo-agents.
  bool has_collision = false;
  running_gamma_id = 1;
  for (size_t i = 0; i < observable_state.exo_agents.size(); i++) {
    vector_t target_velocity = gamma.getAgentVelocity(running_gamma_id);

    // Actuate exo-agent, overriding with observation if necessary.
    float speed = Clamp(target_velocity.norm(), 0.0f, EXO_AGENT_SPEED_MAX);
    float steer = Clamp(1.5f * AngleTo(observable_state.exo_agents[i].heading, target_velocity),
        -AGENT_STEER_MAX, AGENT_STEER_MAX);
    if (observation == nullptr) {
      next_sim.observable_state.exo_agents[i] = Actuate(observable_state.exo_agents[i], speed, EXO_AGENT_ACCEL, steer);
    } else {
      next_sim.observable_state.exo_agents[i] = observation->exo_agents[i];
    }
    log_prob += ActuateLogProb(observable_state.exo_agents[i], speed, EXO_AGENT_ACCEL, steer,
        next_sim.observable_state.exo_agents[i]);

    // Step exo-agent intention.
    std::tuple<carla::sumonetwork::RoutePoint, size_t, float> exo_agent_rp_step = StepRoutePoint(
        next_sim.observable_state.exo_agents[i].position, unobservable_state.exo_agent_intentions[i], EXO_AGENT_PATH_RESOLUTION);
    next_sim.unobservable_state.exo_agent_intentions[i] = std::get<0>(exo_agent_rp_step);

    // Check terminal and collision conditions.
    array_t<vector_t, 4> exo_agent_bb_corners = GetBoundingBoxCorners(next_sim.observable_state.exo_agents[i]);
    if (!has_collision) {
      if (RectangleIntersects(ego_agent_bb_corners, exo_agent_bb_corners)) {
        has_collision = true;
      }
    }

    running_gamma_id++;
  }

  if (has_collision) {
    reward = COLLISION_REWARD;
    next_sim._is_terminal = true;
    next_sim._is_failure = true;
  }
  if (!next_sim._is_terminal) {
    if (!ContainsAll(EGO_SUMO_NETWORK_OCCUPANCY, ego_agent_bb_corners)) {
      reward = COLLISION_REWARD;
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }
  if (!next_sim._is_terminal) {
    vector_t progress_heading =
      vector_t((SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[next_sim.observable_state.ego_agent_progress_index + 1])) -
      SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[next_sim.observable_state.ego_agent_progress_index])).normalized();
    if (next_sim.observable_state.ego_agent.heading.dot(progress_heading) < 0) {
      reward = COLLISION_REWARD;
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }
  if (!next_sim._is_terminal) {
    float progress_reward = static_cast<float>(std::get<1>(ego_agent_rp_step)) * EGO_AGENT_PATH_RESOLUTION;
    float low_speed_penalty = observable_state.ego_agent.speed < LOW_SPEED_THRESHOLD ? LOW_SPEED_PENALTY : 0;
    reward = PROGRESS_REWARD_WEIGHT * progress_reward - low_speed_penalty;

    if (next_sim.observable_state.step == planning_depth_limit) {
      next_sim._is_terminal = true;
    }
  }

  return std::make_tuple(next_sim, reward, next_sim.observable_state, log_prob);
}
template std::tuple<DriveHard, float, DriveHard::Observation, float> DriveHard::Step<false>(
    const DriveHard::Action& action, const DriveHard::Observation* observation) const;
template std::tuple<DriveHard, float, DriveHard::Observation, float> DriveHard::Step<true>(
    const DriveHard::Action& action, const DriveHard::Observation* observation) const;

cv::Mat DriveHard::Render(const list_t<DriveHard>& belief_sims,
    const list_t<Action>& macro_action, const vector_t& macro_action_start) const {

  // x: upward on frame. y: rightward on frame.
  constexpr float RESOLUTION = 0.25f;
  auto to_frame = [&](const vector_t& vector) {
    return cv::Point{
      static_cast<int>((vector.y - (EXO_SPAWN_BOUNDS_MIN.y - 20)) / RESOLUTION),
      static_cast<int>(((EXO_SPAWN_BOUNDS_MAX.x + 20) - vector.x) / RESOLUTION)
    };
  };

  cv::Mat frame(
      static_cast<int>((EXO_SPAWN_BOUNDS_MAX.y - EXO_SPAWN_BOUNDS_MIN.y + 80) / RESOLUTION),
      static_cast<int>((EXO_SPAWN_BOUNDS_MAX.x - EXO_SPAWN_BOUNDS_MIN.x + 80) / RESOLUTION),
      CV_8UC3,
      cv::Scalar(255, 255, 255));

  auto render_agent = [&](const Agent& agent, bool is_ego) {
    vector_t sideward = agent.heading.rotated(PI / 2);

    array_t<vector_t, 4> corners = GetBoundingBoxCorners(agent);
    list_t<list_t<cv::Point>> corners_cv = {{
        to_frame(corners[0]),
        to_frame(corners[1]),
        to_frame(corners[2]),
        to_frame(corners[3])
    }};

    if (!is_ego) {
      array_t<vector_t, 4> gamma_corners = GetGammaBoundingBoxCorners(agent);
      list_t<list_t<cv::Point>> gamma_corners_cv = {{
          to_frame(gamma_corners[0]),
          to_frame(gamma_corners[1]),
          to_frame(gamma_corners[2]),
          to_frame(gamma_corners[3])
      }};
      cv::fillPoly(frame, corners_cv, cv::Scalar(0, 0, 255));
      cv::polylines(frame, gamma_corners_cv, true, cv::Scalar(192, 192, 192));
    } else {
      cv::fillPoly(frame, corners_cv, cv::Scalar(255, 0, 0));
    }
    cv::line(frame,
        to_frame(agent.position + AGENT_MIN.y * sideward),
        to_frame(agent.position + AGENT_MAX.y * sideward),
        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  };

  auto render_occupancy = [&](const carla::occupancy::OccupancyMap& occupancy, const cv::Scalar& color) {
    for (const auto& poly : occupancy.GetPolygons()) {
      list_t<list_t<cv::Point>> ring{list_t<cv::Point>{}};
      for (const carla::geom::Vector2D& p : poly[0]) {
        ring[0].emplace_back(to_frame(p));
      }
      cv::fillPoly(frame, ring, color);
    }
  };

  /*
  for (const auto& occupancy : EXIT_OCCUPANCIES) {
    render_occupancy(occupancy.Difference(JUNCTION_OCCUPANCY), cv::Scalar(0, 0, 255));
  }
  for (const auto& occupancy : ENTRY_OCCUPANCIES) {
    render_occupancy(occupancy.Difference(JUNCTION_OCCUPANCY),cv::Scalar(0, 255, 0));
  }
  */

  render_occupancy(EGO_SUMO_NETWORK_OCCUPANCY, cv::Scalar(240, 240, 240));

  for (const auto& edge : SUMO_NETWORK.Edges()) {
    for (const auto& lane : edge.second.lanes) {
      for (size_t i = 0; i < lane.shape.size() - 1; i++) {
        cv::line(frame, to_frame(lane.shape[i]), to_frame(lane.shape[i + 1]),
            cv::Scalar(192, 192, 192), 1);
      }
    }
  }

  for (size_t i = 0; i < EGO_PATH.size() - 2; i++) {
    cv::line(frame, to_frame(SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[i])), to_frame(SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[i + 1])),
        cv::Scalar(0, 165, 255), 2);
  }

  render_agent(observable_state.ego_agent, true);
  cv::drawMarker(frame,
      to_frame(SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[observable_state.ego_agent_progress_index])),
      cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 10, 2);

  for (size_t i = 0; i < observable_state.exo_agents.size(); i++) {
    render_agent(observable_state.exo_agents[i], false);
  }

  for (const DriveHard& belief_sim : belief_sims) {
    cv::drawMarker(frame, to_frame(belief_sim.observable_state.ego_agent.position), cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 2);
    for (size_t i = 0; i < observable_state.exo_agents.size(); i++) {
      cv::drawMarker(frame,
          to_frame(SUMO_NETWORK.GetRoutePointPosition(
                belief_sim.unobservable_state.exo_agent_intentions[i])),
          cv::Scalar(0, 0, 0), cv::MARKER_CROSS, 10, 2);
    }
  }

  return frame;
}

/* ====== Serialization functions ====== */

vector_t DriveHard::PositionToEgoCentric(const vector_t& p) const {
  vector_t forward = observable_state.ego_agent.heading;
  vector_t sideward = forward.rotated(PI / 2);
  return vector_t(
      (p - observable_state.ego_agent.position).dot(forward),
      (p - observable_state.ego_agent.position).dot(sideward));
}

vector_t DriveHard::HeadingToEgoCentric(const vector_t& h) const {
  return h.rotated(AngleTo(observable_state.ego_agent.heading, vector_t(1, 0))).normalized();
}

void DriveHard::EncodeNonEgoCentric(const Agent& agent, list_t<float>& data) const {
  agent.position.Encode(data);
  agent.heading.Encode(data);
  data.emplace_back(agent.speed);
  data.emplace_back(agent.steer);
}

void DriveHard::EncodeEgoCentric(const Agent& agent, list_t<float>& data) const {
  PositionToEgoCentric(agent.position).Encode(data);
  HeadingToEgoCentric(agent.heading).Encode(data);
  data.emplace_back(agent.speed);
  data.emplace_back(agent.steer);
}

void DriveHard::EncodeEgoCentric(const carla::sumonetwork::RoutePoint& rp, list_t<float>& data) const {
  vector_t position = SUMO_NETWORK.GetRoutePointPosition(rp);
  list_t<carla::sumonetwork::RoutePoint> next_rps = SUMO_NETWORK.GetNextRoutePoints(rp, 1.0f);
  vector_t next_pos = SUMO_NETWORK.GetRoutePointPosition(
      next_rps[std::uniform_int_distribution<size_t>(0, next_rps.size() - 1)(Rng())]);
  vector_t direction = (next_pos - position).normalized();

  PositionToEgoCentric(position).Encode(data);
  HeadingToEgoCentric(direction).Encode(data);
}

void DriveHard::Encode(list_t<float>& data) const {
  EncodeNonEgoCentric(observable_state.ego_agent, data); // 6
  EncodeEgoCentric(EGO_PATH[observable_state.ego_agent_progress_index], data); // 4.
  data.emplace_back(static_cast<float>(observable_state.step)); // 1
  for (size_t i = 0; i < NUM_LOCAL_AGENTS; i++) {
    EncodeEgoCentric(observable_state.exo_agents[i], data); // 6
    EncodeEgoCentric(unobservable_state.exo_agent_intentions[i], data); // 4
  }
}

void DriveHard::EncodeContext(list_t<float>& data) {
  for (size_t i = 0; i < EGO_PATH.size(); i += CONTEXT_PATH_INTERVAL) {
    PositionToEgoCentric(SUMO_NETWORK.GetRoutePointPosition(EGO_PATH[i])).Encode(data);
  }
}

}
