#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/planning/DespotPlanner.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
namespace po = boost::program_options;

typedef planning::DespotPlanner<ExpSimulation> ExpPlanner;

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  size_t macro_length;
  desc.add_options()
      ("macro-length", po::value<size_t>(&macro_length))
      ("visualize", "visualize macro-actions")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  while (true) {
    // Initialize simulation.
    ExpSimulation sim = ExpSimulation::CreateRandom();

    // Initialize belief.
    ExpBelief belief = ExpBelief::FromInitialState(sim);

    // Initialize planner.
    ExpPlanner planner;

    size_t steps = 0;
    while (true) {

      // Write state.
      std::cout << SerializeBelief(belief) << std::endl;

      // Get macro actions.
      list_t<list_t<ExpSimulation::Action>> macro_actions = DeserializeMacroActions(
          get_cin_line(), macro_length);
#if defined SIM_LightDark || defined SIM_IntentionTag
      ExpSimulation::Action trigger_action;
      trigger_action.trigger = true;
      macro_actions.emplace_back();
      macro_actions.back().emplace_back(trigger_action);
#endif

      // Execute planner.
      ExpPlanner::SearchResult search_result = planner.Search(belief, macro_actions);

      // Execute macro-action.
      ExpPlanner::ExecutionResult execution_result = ExpPlanner::ExecuteMacroAction(
          sim, search_result.action, ExpSimulation::MAX_STEPS - steps);
      for (size_t i = 0; i < execution_result.state_trajectory.size(); i++) {
        belief.Update(search_result.action[i], execution_result.observation_trajectory[i]);
        sim = execution_result.state_trajectory[i];
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
      }

      std::cout << search_result.value << std::endl; // Seach max value.
      std::cout << execution_result.state_trajectory.size() << std::endl; // Execution num steps.
      std::cout << execution_result.undiscounted_reward << std::endl; // Execution total undiscounted reward.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS) ? 1 : 0) << std::endl; // Execution terminal.
      std::cout << search_result.num_nodes << std::endl; // Num nodes.
      std::cout << search_result.depth << std::endl; // Search depth.
      std::cout << belief.Error(sim) << std::endl; // Tracking error.

      if (sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS) {
        break;
      }
    }
  }
}

