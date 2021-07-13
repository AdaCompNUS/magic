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
typedef planning::DespotPlanner<ExpSimulation, ExpBelief, true> ExpPlanner;

// Specialize for DriveHard. Particle size should be
// 11 + 6 * NUM_EXO_AGENTS + 4 * #PARTICLES * NUM_EXO_AGENTS
template <>
std::string SerializeBelief(const ExpBelief& belief) {
  list_t<float> values;
  for (size_t i = 0; i < BELIEF_SAMPLES; i++) {
    belief.Sample().Encode(values);
  }
  return macaron::Base64::Encode(ToBytes(values));
}


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

  // Configure planning time based on macro length.
  ExpSimulation::PLANNING_TIME = static_cast<size_t>(
      1000 * (ExpSimulation::DELTA * static_cast<float>(macro_length) - ExpSimulation::BELIEF_UPDATE_ALLOWANCE));

  while (true) {
    // Initialize simulation.
    ExpSimulation sim = ExpSimulation::CreateRandom();

    // Initialize planner.
    ExpPlanner planner;

    // Initialize belief.
    ExpBelief belief(sim);

    size_t steps = 0;
    list_t<ExpSimulation::Action> macro_action(macro_length, {0, 0});

    list_t<float> speed_statistics;
    list_t<float> steer_statistics;
    list_t<float> max_steer_statistics(1, 0.0f);
    list_t<float> speed_penalty_statistics;
    list_t<float> progress_distance_statistics(1, 0.0f);

    while (true) {

      // Forward belief and plan.
      ExpBelief forward_belief = belief;
      bool has_forward_search_result = false;
      ExpPlanner::SearchResult forward_search_result;
      for (const ExpSimulation::Action& a : macro_action) {
        forward_belief = forward_belief.Predict(a);
        if (forward_belief.IsTerminal()) {
          break;
        }
      }


      if (!forward_belief.IsTerminal()) {

        // Write context.
        list_t<float> serialized_context_values;
        forward_belief.Sample().EncodeContext(serialized_context_values);
        std::string serialized_context = macaron::Base64::Encode(ToBytes(serialized_context_values));
        std::cout << serialized_context << std::endl;

        //std::cout << SerializeContext<ExpSimulation>() << std::endl;

        forward_belief = forward_belief.ResampleNonTerminal();
        std::cout << SerializeBelief(forward_belief) << std::endl;
        list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(
            FromBytes<float>(macaron::Base64::Decode(get_cin_line())), macro_length);
        forward_search_result = planner.Search(forward_belief, macro_actions);
        has_forward_search_result = true;
      } else {
        std::cout << std::endl;
        std::cout << std::endl;
      }

      // Execute macro-action.
      ExpPlanner::ExecutionResult execution_result = ExpPlanner::ExecuteMacroAction(
          sim, macro_action, ExpSimulation::MAX_STEPS - steps);
      vector_t macro_action_start = sim.observable_state.ego_agent.position;
      for (size_t i = 0; i < execution_result.state_trajectory.size(); i++) {
        belief.Update(macro_action[i], execution_result.observation_trajectory[i]);

        speed_statistics.emplace_back(std::abs(
              execution_result.state_trajectory[i].observable_state.ego_agent.speed));
        steer_statistics.emplace_back(std::abs(
              execution_result.state_trajectory[i].observable_state.ego_agent.steer));
        max_steer_statistics[0] = std::max(max_steer_statistics[0], steer_statistics.back());
        speed_penalty_statistics.emplace_back(execution_result.state_trajectory[i].observable_state.ego_agent.speed < ExpSimulation::LOW_SPEED_THRESHOLD ? 1 : 0);

        sim = execution_result.state_trajectory[i];
        steps++;

        if (vm.count("visualize")) {
          cv::Mat frame = sim.CreateLocal().Render({}, macro_action, macro_action_start);
          cv::imshow("Frame", frame);
          cv::waitKey(1000 * ExpSimulation::DELTA);
        }
      }

      std::cout << (has_forward_search_result ? forward_search_result.value : 10000000.0f) << std::endl; // Seach max value.
      std::cout << execution_result.state_trajectory.size() << std::endl; // Execution num steps.
      std::cout << execution_result.reward << std::endl; // Execution discounted reward.
      std::cout << execution_result.undiscounted_reward << std::endl; // Execution total undiscounted reward.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || forward_belief.IsTerminal()) ? 1 : 0) << std::endl; // Execution terminal.
      std::cout << (sim.IsFailure() ? 1 : 0) << std::endl; // Failure.
      std::cout << macro_action.size() << std::endl; // Macro action length.
      std::cout << (has_forward_search_result ? forward_search_result.num_nodes : 10000000) << std::endl; // Num nodes.
      std::cout << (has_forward_search_result ? forward_search_result.depth : 10000000) << std::endl; // Search depth.

      if (sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS || forward_belief.IsTerminal()) {
        std::cout << macaron::Base64::Encode(ToBytes(speed_statistics)) << std::endl; // Stat 0.
        std::cout << macaron::Base64::Encode(ToBytes(steer_statistics)) << std::endl; // Stat 1.
        progress_distance_statistics[0] = sim.observable_state.distance;
        std::cout << macaron::Base64::Encode(ToBytes(progress_distance_statistics)) << std::endl; // Stat 2.
        std::cout << macaron::Base64::Encode(ToBytes(max_steer_statistics)) << std::endl; // Stat 3.
        std::cout << macaron::Base64::Encode(ToBytes(speed_penalty_statistics)) << std::endl; // Stat 4.
        break;
      } else {
        std::cout << std::endl; // Stat 0.
        std::cout << std::endl; // Stat 1.
        std::cout << std::endl; // Stat 2.
        std::cout << std::endl; // Stat 3.
        std::cout << std::endl; // Stat 4.
      }

      macro_action = forward_search_result.action;
    }
  }
}

