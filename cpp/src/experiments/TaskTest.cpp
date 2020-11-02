#include "core/Types.h"
#include "core/Util.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
  // Initialize simulation.
  ExpSimulation sim = ExpSimulation::CreateRandom();

  // Initialize belief.
  ExpBelief belief = ExpBelief::FromInitialState(sim);

  while (true) {
    list_t<ExpSimulation> samples;
    for (size_t i = 0; i < 1000; i++) {
      samples.emplace_back(belief.Sample());
    }
    cv::Mat frame = sim.Render(samples);
    cv::imshow("Frame", frame);
    cv::waitKey(1000 * ExpSimulation::DELTA);

    auto action = ExpSimulation::Action::Rand();
    auto result = sim.Step<true>(action);
    belief.Update(action, std::get<2>(result));
    sim = std::get<0>(result);
  }
}
