#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/planning/DespotPlanner.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <macaron/Base64.h>
#include <opencv2/highgui.hpp>
namespace po = boost::program_options;

typedef Belief<ExpSimulation> ExpBelief;
typedef planning::DespotPlanner<ExpSimulation, ExpBelief> ExpPlanner;

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
    ExpBelief belief = ExpBelief::FromInitialState();

    // Initialize planner.
    ExpPlanner planner;

    list_t<float> belief_error_statistics;
    list_t<float> min_belief_error_statistics(1, std::numeric_limits<float>::infinity());

    size_t steps = 0;
    while (true) {

      // Write context.
      std::cout << SerializeContext<ExpSimulation>() << std::endl;

      // Write state.
      std::cout << SerializeBelief(belief) << std::endl;

      // Get macro actions.
      list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(
          FromBytes<float>(macaron::Base64::Decode(get_cin_line())), macro_length);

      // Execute planner.
      ExpPlanner::SearchResult search_result = planner.Search(belief.ResampleNonTerminal(), macro_actions);

      // Execute macro-action.
      ExpPlanner::ExecutionResult execution_result = ExpPlanner::ExecuteMacroAction(
          sim, search_result.action, ExpSimulation::MAX_STEPS - steps);
#ifdef SIM_PuckPush
      vector_t macro_action_start = sim.bot_position;
#else
      vector_t macro_action_start = sim.ego_agent_position;
#endif

      for (size_t i = 0; i < execution_result.state_trajectory.size() && !belief.IsTerminal(); i++) {
        if (vm.count("visualize")) {
          list_t<ExpSimulation> samples;
          for (size_t j = 0; j < 1000; j++) {
            samples.emplace_back(belief.Sample());
          }
          cv::Mat frame = sim.Render(samples, macro_actions, macro_action_start);
          cv::imshow("Frame", frame);
          cv::waitKey(1000 * ExpSimulation::DELTA);
        }

        belief.Update(search_result.action[i], execution_result.observation_trajectory[i]);
        belief_error_statistics.emplace_back(belief.Error(execution_result.state_trajectory[i]));
        min_belief_error_statistics[0] = std::min(min_belief_error_statistics[0], belief_error_statistics.back());
        sim = execution_result.state_trajectory[i];
        steps++;
      }

      std::cout << search_result.value << std::endl; // Seach max value.
      std::cout << execution_result.state_trajectory.size() << std::endl; // Execution num steps.
      std::cout << execution_result.reward << std::endl; // Execution discounted reward.
      std::cout << execution_result.undiscounted_reward << std::endl; // Execution total undiscounted reward.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || belief.IsTerminal()) ? 1 : 0) << std::endl; // Execution terminal.
      std::cout << (sim.IsFailure() ? 1 : 0) << std::endl;
      std::cout << macro_actions[0].size() << std::endl; // Macro action length.
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

