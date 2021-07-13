#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/planning/PomcpowPlanner.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
namespace po = boost::program_options;

typedef Belief<ExpSimulation> ExpBelief;
typedef planning::PomcpowPlanner<ExpSimulation, ExpBelief, true> ExpPlanner;

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  desc.add_options()
      ("ucb", po::value<std::string>()->default_value(""))
      ("k_action", po::value<std::string>()->default_value(""))
      ("alpha_action", po::value<std::string>()->default_value(""))
      ("k_observation", po::value<std::string>()->default_value(""))
      ("alpha_observation", po::value<std::string>()->default_value(""))
      ("visualize", "visualize macro-actions")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  float ucb = vm["ucb"].as<std::string>().empty() ?
      ExpSimulation::POMCPOW_UCB : std::stof(vm["ucb"].as<std::string>());
  float k_action = vm["k_action"].as<std::string>().empty() ?
      ExpSimulation::POMCPOW_K_ACTION : std::stof(vm["k_action"].as<std::string>());
  float alpha_action = vm["alpha_action"].as<std::string>().empty() ?
      ExpSimulation::POMCPOW_ALPHA_ACTION : std::stof(vm["alpha_action"].as<std::string>());
  float k_observation = vm["k_observation"].as<std::string>().empty() ?
      ExpSimulation::POMCPOW_K_OBSERVATION : std::stof(vm["k_observation"].as<std::string>());
  float alpha_observation = vm["alpha_observation"].as<std::string>().empty() ?
      ExpSimulation::POMCPOW_ALPHA_OBSERVATION : std::stof(vm["alpha_observation"].as<std::string>());

  // Planning time after allowance.
  ExpSimulation::PLANNING_TIME = static_cast<size_t>(1000 * (ExpSimulation::DELTA - ExpSimulation::BELIEF_UPDATE_ALLOWANCE));

  while (true) {
    // Initialize simulation.
    ExpSimulation sim = ExpSimulation::CreateRandom();

    // Initialize belief.
    ExpBelief belief(sim);

    size_t steps = 0;
    ExpSimulation::Action current_action = {0, 0};

    list_t<float> speed_statistics;
    list_t<float> steer_statistics;
    list_t<float> max_steer_statistics(1, 0.0f);
    list_t<float> speed_penalty_statistics;
    list_t<float> progress_distance_statistics(1, 0.0f);

    while (true) {

      // Forward belief and plan.
      ExpBelief forward_belief = belief;
      ExpPlanner::SearchResult forward_search_result;
      bool has_forward_search_result = false;
      forward_belief = forward_belief.Predict(current_action);
      if (!forward_belief.IsTerminal()) {
        forward_belief = forward_belief.ResampleNonTerminal();
        forward_search_result = ExpPlanner(ucb, k_action, alpha_action, k_observation, alpha_observation)
            .Search(forward_belief);
        has_forward_search_result = true;
      }

      // Execute planner result.
      std::tuple<ExpSimulation, float, ExpSimulation::Observation, float> step_result = sim.Step<false>(forward_search_result.action);

      // Update belief and sim.
      belief.Update(forward_search_result.action, std::get<2>(step_result));
      sim = std::get<0>(step_result);
      speed_statistics.emplace_back(std::abs(sim.observable_state.ego_agent.speed));
      steer_statistics.emplace_back(std::abs(sim.observable_state.ego_agent.steer));
      max_steer_statistics[0] = std::max(max_steer_statistics[0], steer_statistics.back());
      speed_penalty_statistics.emplace_back(sim.observable_state.ego_agent.speed < ExpSimulation::LOW_SPEED_THRESHOLD ? 1 : 0);
      steps++;

     if (vm.count("visualize")) {
        list_t<ExpSimulation> samples;
        for (size_t i = 0; i < 1000; i++) {
          samples.emplace_back(belief.Sample());
        }
        cv::Mat frame = sim.Render(samples);
        cv::imshow("Frame", frame);
        cv::waitKey(1000 * ExpSimulation::DELTA);
      }

      std::cout << (has_forward_search_result ? forward_search_result.value : 10000000.0f) << std::endl; // Seach max value.
      std::cout << 1 << std::endl; // Execution num steps.
      std::cout << std::get<1>(step_result) << std::endl; // Execution total undiscounted reward.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || forward_belief.IsTerminal()) ? 1 : 0) << std::endl; // Execution terminal.
      std::cout << (sim.IsFailure() ? 1 : 0) << std::endl; // Failure.
      std::cout << (has_forward_search_result ? forward_search_result.num_nodes : 10000000) << std::endl; // Num nodes.
      std::cout << (has_forward_search_result ? forward_search_result.depth : 10000000) << std::endl; // Search depth.

      if ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || forward_belief.IsTerminal()) ? 1 : 0) {
        std::cout << macaron::Base64::Encode(ToBytes(speed_statistics)) << std::endl; // Stat 0.
        std::cout << macaron::Base64::Encode(ToBytes(steer_statistics)) << std::endl; // Stat 2.
        progress_distance_statistics[0] = sim.observable_state.distance;
        std::cout << macaron::Base64::Encode(ToBytes(progress_distance_statistics)) << std::endl; // Stat 3.
        std::cout << macaron::Base64::Encode(ToBytes(max_steer_statistics)) << std::endl; // Stat 4.
        std::cout << macaron::Base64::Encode(ToBytes(speed_penalty_statistics)) << std::endl; // Stat 1.
      } else {
        std::cout << std::endl; // Stat 0.
        std::cout << std::endl; // Stat 1.
        std::cout << std::endl; // Stat 2.
        std::cout << std::endl; // Stat 3.
        std::cout << std::endl; // Stat 4.
      }

      if (sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || forward_belief.IsTerminal()) {
        break;
      }

    }
  }
}

