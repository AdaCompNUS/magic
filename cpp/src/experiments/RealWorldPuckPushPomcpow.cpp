#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/Util.h"
#include "core/planning/PomcpowPlanner.h"
#include "core/simulations/PuckPush.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <boost/program_options.hpp>
#include <future>
#include <iostream>
#include <macaron/Base64.h>
#include <stdexcept>
namespace po = boost::program_options;

typedef Belief<ExpSimulation> ExpBelief;
typedef planning::PomcpowPlanner<ExpSimulation, ExpBelief> ExpPlanner;

std::string ToState(const ExpBelief& belief) {
  list_t<float> values;
  for (size_t i = 0; i < BELIEF_SAMPLES; i++) {
    belief.Sample().Encode(values);
  }
  return macaron::Base64::Encode(ToBytes(values));
}

void RwiPuckPush(size_t macro_length, bool handcrafted) {

  // Initialize belief.
  ExpSimulation::CreateRandom();
  ExpBelief belief = ExpBelief::FromInitialState();
  std::future<ExpPlanner::SearchResult> planner_future;

  // Initialize planner.
  ExpPlanner planner;

  while (true) {

    std::string instruction = get_cin_line();

    if (instruction == "INITIALIZE_BELIEF") {
      ExpSimulation sim;
      sim.bot_position.x = std::stof(get_cin_line());
      sim.bot_position.y = std::stof(get_cin_line());

      std::string puck_x_str = get_cin_line();
      if (puck_x_str == "") {
        sim.puck_position.x = std::numeric_limits<float>::quiet_NaN();
      } else {
        sim.puck_position.x = std::stof(puck_x_str);
      }

      std::string puck_y_str = get_cin_line();
      if (puck_y_str == "") {
        sim.puck_position.y = std::numeric_limits<float>::quiet_NaN();
      } else {
        sim.puck_position.y = std::stof(puck_y_str);
      }

      belief = ExpBelief::FromInitialState(sim);
      std::cout << std::endl << std::flush; // Send reply to signal completion.
    } else if (instruction == "UPDATE_BELIEF") {
      float action_f = std::stof(get_cin_line());
      ExpSimulation::Action action{action_f};

      ExpSimulation::Observation observation;
      observation.bot_position.x = std::stof(get_cin_line());
      observation.bot_position.y = std::stof(get_cin_line());

      std::string puck_x_str = get_cin_line();
      if (puck_x_str == "") {
        observation.puck_position.x = std::numeric_limits<float>::quiet_NaN();
      } else {
        observation.puck_position.x = std::stof(puck_x_str);
      }

      std::string puck_y_str = get_cin_line();
      if (puck_y_str == "") {
        observation.puck_position.y = std::numeric_limits<float>::quiet_NaN();
      } else {
        observation.puck_position.y = std::stof(puck_y_str);
      }

      belief.Update(action, observation);

      std::cout << std::endl << std::flush; // Send reply to signal completion.
    } else if (instruction == "SAMPLE_BELIEF") {
      size_t count = static_cast<size_t>(std::stoi(get_cin_line()));
      for (size_t i = 0; i < count; i++) {
        ExpSimulation sim = belief.Sample();
        std::cout << sim.bot_position.x << std::endl;
        std::cout << sim.bot_position.y << std::endl;
        std::cout << sim.puck_position.x << std::endl;
        std::cout << sim.puck_position.y << std::endl;
      }
      std::cout << std::flush;
    } else if (instruction == "GET_BELIEF") {
      std::cout << ToState(belief) << std::endl;
    } else if (instruction == "INVOKE_PLANNER") {
      std::string sgpc_params;
      std::getline(std::cin, sgpc_params); // Macro-actions. Skipped.

      ExpPlanner::SearchResult search_result = planner.Search(belief.ResampleNonTerminal());

      std::cout << search_result.action.orientation << std::endl;
      std::cout << search_result.num_nodes << std::endl;
      std::cout << search_result.depth << std::endl;
      std::cout << search_result.value << std::endl;
    } else if (instruction == "SHUTDOWN") {
      break;
    }
  }
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  size_t macro_length;
  desc.add_options()
      ("macro-length", po::value<size_t>(&macro_length))
      ("handcrafted", "use handcrafted macro-actions")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  RwiPuckPush(macro_length, vm.count("handcrafted"));
}
