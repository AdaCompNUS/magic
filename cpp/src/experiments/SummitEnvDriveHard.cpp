#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/planning/DespotPlanner.h"
#include "core/planning/PomcpowPlanner.h"
#include "core/simulations/DriveHard.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <macaron/Base64.h>
#include <opencv2/highgui.hpp>
#include <future>
#include <stdexcept>
namespace po = boost::program_options;

typedef Belief<ExpSimulation> ExpBelief;
typedef planning::DespotPlanner<ExpSimulation, ExpBelief, true> ExpDespotPlanner;
typedef planning::PomcpowPlanner<ExpSimulation, ExpBelief, true> ExpPomcpowPlanner;

inline static ExpSimulation::Action read_action() {
  list_t<float> action_data = FromBytes<float>(macaron::Base64::Decode(get_cin_line()));
  ExpSimulation::Action action;
  action.speed = action_data[0];
  action.steer = action_data[1];
  return action;
}

inline static list_t<ExpSimulation::Action> read_macro_action(size_t macro_length) {
  list_t<ExpSimulation::Action> macro_action;
  for (size_t i = 0; i < macro_length; i++) {
    macro_action.emplace_back(read_action());
  }
  return macro_action;
}

inline static ExpSimulation::Observation read_observation() {
  list_t<float> state_data = FromBytes<float>(macaron::Base64::Decode(get_cin_line()));
  ExpSimulation::Observation obs;
  obs.ego_agent.position.x = state_data[0];
  obs.ego_agent.position.y = state_data[1];
  obs.ego_agent.heading.x = state_data[2];
  obs.ego_agent.heading.y = state_data[3];
  obs.ego_agent.speed = state_data[4];
  obs.ego_agent.steer = state_data[5];
  for (size_t i = 0; i < ExpSimulation::NUM_EXO_AGENTS; i++) {
    obs.exo_agents.emplace_back();
    obs.exo_agents[i].position.x = state_data[6 + 6 * i + 0];
    obs.exo_agents[i].position.y = state_data[6 + 6 * i + 1];
    obs.exo_agents[i].heading.x = state_data[6 + 6 * i + 2];
    obs.exo_agents[i].heading.y = state_data[6 + 6 * i + 3];
    obs.exo_agents[i].speed = state_data[6 + 6 * i + 4];
    obs.exo_agents[i].steer = state_data[6 + 6 * i + 5];
  }
  return obs;
}

inline static list_t<vector_t> read_context() {
  list_t<float> state_data = FromBytes<float>(macaron::Base64::Decode(get_cin_line()));
  list_t<vector_t> path_pos;
  for (size_t i = 0; i < ExpSimulation::EGO_PATH.size(); i++) {
    path_pos.emplace_back(state_data[2 * i], state_data[2 * i + 1]);
  }
  return path_pos;
}

inline static std::string SerializeMacroAction(const list_t<ExpSimulation::Action>& macro_action) {
  list_t<float> macro_action_data;
  for (const ExpSimulation::Action& action : macro_action) {
    macro_action_data.emplace_back(action.speed);
    macro_action_data.emplace_back(action.steer);
  }
  return macaron::Base64::Encode(ToBytes(macro_action_data));
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  size_t macro_length;
  desc.add_options()
      ("macro-length", po::value<size_t>(&macro_length))
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Configure planning time based on macro length.
  ExpSimulation::PLANNING_TIME = static_cast<size_t>(
      1000 * (ExpSimulation::DELTA * static_cast<float>(macro_length) - ExpSimulation::BELIEF_UPDATE_ALLOWANCE));

  // Hand-crafted macro-actions.
  list_t<list_t<ExpSimulation::Action>> hc_macro_actions = ExpSimulation::Action::CreateHandcrafted(macro_length);

  // Declare belief.
  ExpBelief belief;

  std::future<ExpBelief> update_belief_task;
  std::future<std::pair<list_t<ExpSimulation::Action>, size_t>> forward_plan_task;

  list_t<std::string> log;

  bool has_cv = false;
  while (true) {
    std::string instruction = get_cin_line();
    log.emplace_back(std::to_string(belief.Sample().observable_state.step) + ": " + instruction);
    //std::cerr << "[CPP][" << std::to_string(belief.Sample().observable_state.step) << ": " << instruction << "]" << std::endl;

    if (instruction == "INITIALIZE_BELIEF") {

      // Convert obs to sim.
      ExpSimulation::Observation obs = read_observation();
      ExpSimulation sim;
      sim.observable_state = ExpSimulation::ObservableState::From(obs);
      sim.observable_state.ego_agent_progress_index = 0;
      for (size_t i = 0; i < ExpSimulation::NUM_EXO_AGENTS; i++) {
        sim.unobservable_state.exo_agent_intentions.emplace_back(ExpSimulation::SUMO_NETWORK.GetNearestRoutePoint(
            sim.observable_state.exo_agents[i].position));
      }

      list_t<vector_t> context = read_context();
      ExpSimulation::CreateRandom(context);

      belief = ExpBelief(sim);
      std::cout << std::endl;

    } else if (instruction == "UPDATE_BELIEF") {

      // Reads no matter what.
      ExpSimulation::Action action = read_action();
      ExpSimulation::Observation observation = read_observation();

      if (update_belief_task.valid()) {
        throw std::logic_error("Previous update belief task still exists!");
      }

      if (belief.IsTerminal()) {
        throw std::logic_error("Belief is already terminal!");
      }

      update_belief_task = std::async(std::launch::async,
          [](const ExpBelief& belief, const ExpSimulation::Action& action, const ExpSimulation::Observation& observation) {
            ExpBelief updated_belief = belief;
            updated_belief.Update(action, observation);
            return updated_belief;
      }, belief, action, observation);
      std::cout << std::endl;

    } else if (instruction == "POP_UPDATE_BELIEF") {

      if (!update_belief_task.valid()) {
        throw std::logic_error("Update belief task non-existent!");
      }

      belief = update_belief_task.get();
      update_belief_task = {};

      if (belief.IsTerminal()) {
        std::cout << "TERMINAL" << std::endl;
      } else {
        std::cout << std::endl;
      }
    } else if (instruction == "SAMPLE_BELIEF") {
      std::cout << SerializeBelief(belief) << std::endl;
    } else if (instruction == "FORWARD_PLAN_HANDCRAFTED") {

      list_t<ExpSimulation::Action> forward_macro_action = read_macro_action(macro_length);

      if (forward_plan_task.valid()) {
        throw std::logic_error("Previous forward plan task still exists!");
      }

      if (belief.IsTerminal()) {
        throw std::logic_error("Belief is already terminal!");
      }

      // Forward belief.
      ExpBelief forward_belief = belief;
      for (const ExpSimulation::Action& a : forward_macro_action) {
        forward_belief = forward_belief.Predict(a);
        if (forward_belief.IsTerminal()) {
          break;
        }
      }

      if (forward_belief.IsTerminal()) {
        forward_plan_task = std::async(std::launch::async, []() { return std::pair<list_t<ExpSimulation::Action>, size_t>{}; });
        std::cout << "TERMINAL" << std::endl;
      } else {
        // Start forward plan task.
        forward_plan_task = std::async(std::launch::async,
            [](const ExpBelief& forward_belief, const list_t<list_t<ExpSimulation::Action>>& macro_actions) {

              // Plan against forwarded belief.
              ExpDespotPlanner::SearchResult search_result = ExpDespotPlanner().Search(
                  forward_belief.ResampleNonTerminal(), macro_actions);
              return std::make_pair(search_result.action, search_result.depth);

            }, forward_belief, hc_macro_actions);

        std::cout << std::endl;
      }

    } else if (instruction == "FORWARD_PLAN_POMCPOW") {

      list_t<ExpSimulation::Action> forward_macro_action = read_macro_action(macro_length);

      if (forward_macro_action.size() != 1) {
        throw std::logic_error("Only primitive actions can be used!");
      }

      if (forward_plan_task.valid()) {
        throw std::logic_error("Previous forward plan task still exists!");
      }

      if (belief.IsTerminal()) {
        throw std::logic_error("Belief is already terminal!");
      }

      // Forward belief.
      ExpBelief forward_belief = belief.Predict(forward_macro_action[0]);

      if (forward_belief.IsTerminal()) {
        forward_plan_task = std::async(std::launch::async, []() { return std::pair<list_t<ExpSimulation::Action>, size_t>{}; });
        std::cout << "TERMINAL" << std::endl;
      } else {
        // Start forward plan task.
        forward_plan_task = std::async(std::launch::async,
            [](const ExpBelief& forward_belief) {

              // Plan against forwarded belief.
              ExpPomcpowPlanner::SearchResult search_result = ExpPomcpowPlanner().Search(
                  forward_belief.ResampleNonTerminal());
              return std::make_pair(list_t<ExpSimulation::Action>{search_result.action}, search_result.depth);

            }, forward_belief);

        std::cout << std::endl;
      }

    } else if (instruction == "FORWARD_PLAN_MAGIC") {

      list_t<ExpSimulation::Action> forward_macro_action = read_macro_action(macro_length);

      if (forward_plan_task.valid()) {
        throw std::logic_error("Previous forward plan task still exists!");
      }

      // Forward belief.
      ExpBelief forward_belief = belief;
      for (const ExpSimulation::Action& a : forward_macro_action) {
        forward_belief = forward_belief.Predict(a);
        if (forward_belief.IsTerminal()) {
          break;
        }
      }

      if (forward_belief.IsTerminal()) {
        forward_plan_task = std::async(std::launch::async, []() { return std::pair<list_t<ExpSimulation::Action>, size_t>{}; });
        std::cout << "TERMINAL" << std::endl;
      } else {
        forward_belief = forward_belief.ResampleNonTerminal();
        std::cout << std::endl; // Write successs.

        // Write context.
        list_t<float> serialized_context_values;
        forward_belief.Sample().EncodeContext(serialized_context_values);
        std::string serialized_context = macaron::Base64::Encode(ToBytes(serialized_context_values));
        std::cout << serialized_context << std::endl;
        std::cout << SerializeBelief(forward_belief) << std::endl; // Write forward belief.

        // Read params
        list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(
              FromBytes<float>(macaron::Base64::Decode(get_cin_line())), macro_length);

        // Start forward plan task.
        forward_plan_task = std::async(std::launch::async,
            [](const ExpBelief& forward_belief, const list_t<list_t<ExpSimulation::Action>>& macro_actions) {

                // Plan against forwarded belief.
                ExpDespotPlanner::SearchResult search_result = ExpDespotPlanner().Search(forward_belief, macro_actions);
                return std::make_pair(search_result.action, search_result.depth);

            }, forward_belief, macro_actions);

        std::cout << std::endl;
      }

    } else if (instruction == "POP_FORWARD_PLAN_RESULT") {
      if (!forward_plan_task.valid()) {
        throw std::logic_error("Forward plan task non-existent!");
      }
      std::pair<list_t<ExpSimulation::Action>, size_t> result = forward_plan_task.get();
      if (std::get<0>(result).size() == 0) {
        std::cout << "TERMINAL" << std::endl;
      } else {
        std::cout << SerializeMacroAction(std::get<0>(result)) << std::endl;
        std::cout << std::get<1>(result) << std::endl;
      }
      forward_plan_task = {};
    } else if (instruction == "DEBUG_STATE") {

    } else if (instruction == "DEBUG_BELIEF") {

      // Convert obs to sim.
      ExpSimulation::Observation obs = read_observation();
      ExpSimulation sim;
      sim.observable_state = ExpSimulation::ObservableState::From(obs);
      sim.observable_state.ego_agent_progress_index = 0;
      for (size_t i = 0; i < ExpSimulation::NUM_EXO_AGENTS; i++) {
        sim.unobservable_state.exo_agent_intentions.emplace_back(ExpSimulation::SUMO_NETWORK.GetNearestRoutePoint(
            sim.observable_state.exo_agents[i].position));
      }

      list_t<ExpSimulation> samples;
      for (size_t i = 0; i < 100; i++) {
        samples.emplace_back(belief.Sample());
      }
      cv::Mat frame = sim.Render(samples);
      cv::imshow("Frame", frame);
      if (!has_cv) {
        cv::moveWindow("Frame", 0, 0);
        has_cv = true;
      }
      cv::waitKey(1);
      std::cout << std::endl;

    } else if (instruction == "TERMINATE") {
      break;
    } else {
      throw std::range_error("Unknonw instruction: " + instruction);
    }
  }
}

