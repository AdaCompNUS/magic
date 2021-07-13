#pragma once

#include "core/Types.h"
#include "core/Util.h"
#include <algorithm>
#include <cstddef>
#include <random>
#include <omp.h>

template <typename T>
class Belief {

public:

  Belief() : _particles(T::BELIEF_NUM_PARTICLES) { }
  Belief(const T& particle) : _particles(T::BELIEF_NUM_PARTICLES, particle) { }

  static Belief FromInitialState() {
    Belief belief;
    for (size_t i = 0; i < T::BELIEF_NUM_PARTICLES; i++) {
      belief._particles[i] = T::SampleBeliefPrior();
    }
    return belief;
  }

  T Sample() const {
    static std::uniform_int_distribution<size_t> dist(0, _particles.size() - 1);
    return _particles[dist(Rng())];
  }

  float Error(const T& state, size_t num_samples=100) const {
    float error = 0;
    for (size_t i = 0; i < num_samples; i++) {
      error += Sample().Error(state);
    }
    return error / static_cast<float>(num_samples);
  }

  bool IsTerminal() const {
    for (const T& particle : _particles) {
      if (!particle.IsTerminal()) {
        return false;
      }
    }
    return true;
  }

  Belief ResampleNonTerminal() const {
    list_t<T> non_terminal_particles;

    for (size_t i = 0; i < _particles.size(); i++) {
      if (!_particles[i].IsTerminal()) {
        non_terminal_particles.emplace_back(_particles[i]);
      }
    }

    // Resample
    Belief new_belief;
    std::uniform_int_distribution<size_t> distribution(0, non_terminal_particles.size() - 1);
    for (size_t i = 0; i < _particles.size(); i++) {
      new_belief._particles[i] = non_terminal_particles[distribution(Rng())];
    }

    return new_belief;
  }

  void Update(const typename T::Action& action, const typename T::Observation& observation) {
    list_t<T> updated_particles;
    list_t<float> weights;

    for (size_t i = 0; i < _particles.size(); i++) {
      if (!_particles[i].IsTerminal()) {
        std::tuple<T, float, typename T::Observation, float> step_result =
            _particles[i].template Step<true>(action, &observation);
        {
          updated_particles.emplace_back(std::get<0>(step_result));
          weights.emplace_back(std::get<3>(step_result));
        }
      }
    }

    weights = SoftMax(weights);

    // Resample
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    for (size_t i = 0; i < _particles.size(); i++) {
      _particles[i] = updated_particles[distribution(Rng())];
    }
  }

  Belief Predict(const typename T::Action& action) {
    list_t<T> updated_particles;

    for (size_t i = 0; i < _particles.size(); i++) {
      if (!_particles[i].IsTerminal()) {
        std::tuple<T, float, typename T::Observation, float> step_result =
            _particles[i].template Step<false>(action);
        {
          updated_particles.emplace_back(std::get<0>(step_result));
        }
      }
    }

    // Resample
    Belief new_belief;
    std::uniform_int_distribution<size_t> distribution(0, updated_particles.size() - 1);
    for (size_t i = 0; i < _particles.size(); i++) {
      new_belief._particles[i] = updated_particles[distribution(Rng())];
    }

    return new_belief;
  }

  void Encode(list_t<float>& data) const {
    for (const T& particle : _particles) {
      particle.Encode(data);
    }
  }

  list_t<T> _particles;

};
