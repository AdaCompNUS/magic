#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "core/Util.h"
#include "core/planning/DespotPlanner.h"
#include "core/simulations/PuckPush.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <boost/program_options.hpp>
#include <future>
#include <iostream>
#include <stdexcept>
namespace po = boost::program_options;

typedef Belief<ExpSimulation> ExpBelief;
typedef planning::DespotPlanner<ExpSimulation, ExpBelief> ExpPlanner;

void RwiPuckPushDespot(size_t macro_length, bool handcrafted) {

  // Hand-crafted macro-actions.
  list_t<list_t<ExpSimulation::Action>> hc_macro_actions;
  for (size_t i = 0; i < 8; i++) {
    hc_macro_actions.emplace_back();
    for (size_t j = 0; j < macro_length; j++) {
      hc_macro_actions.back().push_back({static_cast<float>(i) * PI / 4});
    }
  }

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
      std::cout << SerializeBelief(belief) << std::endl;
    } else if (instruction == "INVOKE_PLANNER") {
      list_t<list_t<ExpSimulation::Action>> macro_actions = ExpSimulation::Action::Deserialize(
          FromBytes<float>(macaron::Base64::Decode(get_cin_line())));
      if (handcrafted) {
        macro_actions = hc_macro_actions;
      }

      ExpPlanner::SearchResult search_result = planner.Search(belief.ResampleNonTerminal(), macro_actions);

      for (const ExpSimulation::Action& action : search_result.action) {
        std::cout << action.orientation << std::endl;
      }
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

  RwiPuckPushDespot(macro_length, vm.count("handcrafted"));
}
