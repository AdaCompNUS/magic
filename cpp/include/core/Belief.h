#pragma once

#include "core/Types.h"
#include "core/Util.h"
#include <algorithm>
#include <cstddef>
#include <random>

template <typename T>
class Belief {

public:

  Belief() : _particles(T::BELIEF_NUM_PARTICLES) { }

  static Belief FromInitialState(const T& state) {
    Belief belief;
    for (size_t i = 0; i < T::BELIEF_NUM_PARTICLES; i++) {
      belief._particles[i] = T::SampleBeliefPrior();
    }
    return belief;
  }

  T Sample() const {
    static std::uniform_int_distribution<size_t> dist(0, T::BELIEF_NUM_PARTICLES - 1);
    return _particles[dist(Rng())];
  }

  float Error(const T& state, size_t num_samples=100) const {
    float error = 0;
    for (size_t i = 0; i < num_samples; i++) {
      error += Sample().Error(state);
    }
    return error / static_cast<float>(num_samples);
  }

  void Update(const typename T::Action& action, const typename T::Observation& observation) {
    list_t<T> updated_particles(T::BELIEF_NUM_PARTICLES);
    list_t<float> weights(T::BELIEF_NUM_PARTICLES);

    //#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < T::BELIEF_NUM_PARTICLES; i++) {
      if (_particles[i].IsTerminal()) {
        weights[i] = -std::numeric_limits<float>::infinity();
      } else {
        std::tuple<T, float, typename T::Observation, float> step_result =
            _particles[i].template Step<true>(action, &observation);
        updated_particles[i] = std::get<0>(step_result);
        weights[i] = std::get<3>(step_result);
      }
    }

    weights = SoftMax(weights);

    // Resample
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    for (size_t i = 0; i < T::BELIEF_NUM_PARTICLES; i++) {
      _particles[i] = updated_particles[distribution(Rng())];
    }
  }

  void Encode(list_t<float>& data) const {
    for (const T& particle : _particles) {
      particle.Encode(data);
    }
  }

private:

  list_t<T> _particles;

};
