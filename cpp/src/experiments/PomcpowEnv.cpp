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
typedef planning::PomcpowPlanner<ExpSimulation, ExpBelief> ExpPlanner;

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

  while (true) {
    // Initialize simulation.
    ExpSimulation sim = ExpSimulation::CreateRandom();

    // Initialize belief.
    ExpBelief belief = ExpBelief::FromInitialState();

    // Initialize planner.
    ExpPlanner planner(ucb, k_action, alpha_action, k_observation, alpha_observation);

    list_t<float> belief_error_statistics;
    list_t<float> min_belief_error_statistics(1, std::numeric_limits<float>::infinity());

    size_t steps = 0;
    while (true) {

      // Execute planner.
      ExpPlanner::SearchResult search_result = planner.Search(belief.ResampleNonTerminal());

      // Execute planner result.
      std::tuple<ExpSimulation, float, ExpSimulation::Observation, float> step_result = sim.Step<false>(search_result.action);

      if (vm.count("visualize")) {
        list_t<ExpSimulation> samples;
        for (size_t i = 0; i < 1000; i++) {
          samples.emplace_back(belief.Sample());
        }
        cv::Mat frame = sim.Render(samples);
        cv::imshow("Frame", frame);
        cv::waitKey(1000 * ExpSimulation::DELTA);
      }

      // Update belief and sim.
      belief.Update(search_result.action, std::get<2>(step_result));
      belief_error_statistics.emplace_back(belief.Error(std::get<0>(step_result)));
      min_belief_error_statistics[0] = std::min(min_belief_error_statistics[0], belief_error_statistics.back());
      sim = std::get<0>(step_result);
      steps++;

      std::cout << search_result.value << std::endl; // Seach max value.
      std::cout << 1 << std::endl; // Execution num steps.
      std::cout << std::get<1>(step_result) << std::endl; // Execution total undiscounted reward.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || belief.IsTerminal()) ? 1 : 0) << std::endl; // Execution terminal.
      std::cout << (sim.IsFailure() ? 1 : 0) << std::endl; // Failure.
      std::cout << search_result.num_nodes << std::endl; // Num nodes.
      std::cout << search_result.depth << std::endl; // Search depth.

      if (sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || belief.IsTerminal()) {
        std::cout << macaron::Base64::Encode(ToBytes(belief_error_statistics)) << std::endl; // Stat 0.
        std::cout << macaron::Base64::Encode(ToBytes(min_belief_error_statistics)) << std::endl; // Stat 1.
        std::cout << std::endl; // Stat 2.
        std::cout << std::endl; // Stat 3.
        std::cout << std::endl; // Stat 4.
        break;
      } else {
        std::cout << std::endl; // Stat 0.
        std::cout << std::endl; // Stat 1.
        std::cout << std::endl; // Stat 2.
        std::cout << std::endl; // Stat 3.
        std::cout << std::endl; // Stat 4.
      }
    }
  }
}
