#include "core/Types.h"
#include "core/Util.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <algorithm>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
  // Initialize simulation.
  ExpSimulation sim = ExpSimulation::CreateRandom();

  // Initialize belief.
  ExpBelief belief = ExpBelief::FromInitialState(sim);

  // Initialize num steps.
  size_t steps = 0;

  while (true) {

    std::string instruction = get_cin_line();

    if (instruction == "RESET") {
      // Reset.
      sim = ExpSimulation::CreateRandom();
      belief = ExpBelief::FromInitialState(sim);
      steps = 0;
      // Write state.
      std::cout << SerializeBelief(belief) << std::endl;
    } else if (instruction == "STEP") {
      float action_x = std::stof(get_cin_line());
      float action_y = std::stof(get_cin_line());
      ExpSimulation::Action action{atan2f(action_y, action_x)};
#if defined SIM_LightDark || defined SIM_IntentionTag
      action.trigger = std::bernoulli_distribution(std::stof(get_cin_line()))(Rng());
#endif

      // Execute action.
      std::tuple<ExpSimulation, float, ExpSimulation::Observation, float> step_result = sim.Step<false>(action);

      // Update belief and sim.
      belief.Update(action, std::get<2>(step_result));
      sim = std::get<0>(step_result);
      steps++;

      // Write state.
      std::cout << SerializeBelief(belief) << std::endl;

      // Write reward.
      std::cout << std::get<1>(step_result) << std::endl;

      // Write is_terminal.
      std::cout << ((sim.IsTerminal() || steps >= ExpSimulation::MAX_STEPS) ? 1 : 0) << std::endl; // Execution terminal.

      // Write additional info.
      std::cout << belief.Error(sim) << std::endl; // Tracking error.
    }
  }
}

