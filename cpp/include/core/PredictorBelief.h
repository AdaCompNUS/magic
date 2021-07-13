#pragma once

#include "core/Types.h"
#include "core/Util.h"
#include <algorithm>
#include <cstddef>
#include <random>

template <typename T>
class PredictorBelief {

public:

  PredictorBelief() {}
  PredictorBelief(const T& particle) : _particles(T::BELIEF_NUM_PARTICLES, particle) { }

  T Sample() const {
    static std::uniform_int_distribution<size_t> dist(0, _particles.size() - 1);
    return _particles[dist(Rng())];
  }

  bool IsTerminal() const {
    for (const T& particle : _particles) {
      if (!particle.IsTerminal()) {
        return false;
      }
    }
    return true;
  }

  void Predict(const typename T::Action& action) {
    list_t<T> updated_particles;

    for (size_t i = 0; i < _particles.size(); i++) {
      if (!_particles[i].IsTerminal()) {
        std::tuple<T, float, typename T::Observation, float> step_result =
            _particles[i].template Step<false>(action);
        updated_particles.emplace_back(std::get<0>(step_result));
      }
    }

    // Resample
    std::uniform_int_distribution<size_t> distribution(0, updated_particles.size() - 1);
    for (size_t i = 0; i < _particles.size(); i++) {
      _particles[i] = updated_particles[distribution(Rng())];
    }
  }

  void Encode(list_t<float>& data) const {
    for (const T& particle : _particles) {
      particle.Encode(data);
    }
  }

  list_t<T> _particles;

};
