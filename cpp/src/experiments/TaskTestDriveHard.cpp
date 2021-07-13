#include "core/Belief.h"
#include "core/Types.h"
#include "core/Util.h"
#include "experiments/Config.h"
#include "experiments/Util.h"
#include <opencv2/highgui.hpp>
#include <chrono>

typedef Belief<ExpSimulation> ExpBelief;

int main(int argc, char** argv) {
  // Initialize simulation.
  ExpSimulation sim = ExpSimulation::CreateRandom();

  // Initialize belief.
  ExpBelief belief(sim);

  while (!sim.IsTerminal()) {
    list_t<ExpSimulation> samples;
    for (size_t i = 0; i < 1000; i++) {
      samples.emplace_back(belief.Sample());
    }
    cv::Mat frame = sim.Render(samples);
    cv::imshow("Frame", frame);
    cv::waitKey(1000 * ExpSimulation::DELTA);

    auto action = ExpSimulation::Action(0, 0);
    auto result = sim.Step<false>(action);
    auto start = std::chrono::system_clock::now();
    belief.Update(action, std::get<2>(result));
    auto end = std::chrono::system_clock::now();
    sim = std::get<0>(result);
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
  }
}
