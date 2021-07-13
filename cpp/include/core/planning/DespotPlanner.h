#pragma once

#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/globals.h>
#include <despot/interface/pomdp.h>
#include <despot/solver/despot.h>
#include <despot/util/logging.h>

namespace planning {

template <typename T, typename TBelief, bool local=false>
class DespotPlanner {
public:

  struct ExecutionResult {
    list_t<T> state_trajectory;
    float reward;
    float undiscounted_reward;
    list_t<typename T::Observation> observation_trajectory;
  };

  struct SearchResult {
    list_t<typename T::Action> action;
    size_t num_nodes;
    size_t depth;
    float value;
  };

  DespotPlanner() { }

  // Returns: State trajectory, cumulative discounted reward, observation trajectory, log_prob.
  static ExecutionResult ExecuteMacroAction(const T& sim,
      const list_t<typename T::Action>& macro_action,
      size_t max_steps=std::numeric_limits<size_t>::max()) {

    T local_sim = sim;

    list_t<T> sims;
    list_t<typename T::Observation> new_observations;

    size_t i = 0;
    float discount = 1.0;
    float reward = 0.0;
    float undiscounted_reward = 0.0;

    for (; i < std::min(max_steps, macro_action.size()) && !local_sim.IsTerminal(); i++) {
      std::tuple<T, float, typename T::Observation, float> step_result =
          local_sim.template Step<false>(macro_action[i]);

      local_sim = std::get<0>(step_result);
      reward += discount * std::get<1>(step_result);
      undiscounted_reward += std::get<1>(step_result);
      discount *= T::GAMMA;

      sims.emplace_back(std::get<0>(step_result));
      new_observations.emplace_back(std::get<2>(step_result));
    }

    return ExecutionResult{sims, reward, undiscounted_reward, new_observations};
  }

  SearchResult Search(
      const TBelief& belief,
      const list_t<list_t<typename T::Action>>& macro_actions) const {
    despot::logging::level(0);
    despot::Globals::config.search_depth = static_cast<int>(lround(ceilf(T::SEARCH_DEPTH / static_cast<float>(macro_actions[0].size()))));
    despot::Globals::config.sim_len = despot::Globals::config.search_depth;
    despot::Globals::config.max_policy_sim_len = despot::Globals::config.search_depth;
    despot::Globals::config.num_scenarios = T::DESPOT_NUM_SCENARIOS;
    despot::Globals::config.time_per_move = static_cast<float>(T::PLANNING_TIME) / 1000;
    despot::Globals::config.discount = powf(T::GAMMA, static_cast<float>(macro_actions[0].size()));
    despot::Globals::config.xi = 0.95;

    DespotModel model(&belief, macro_actions);
    despot::DESPOT solver(
        &model,
        model.CreateScenarioLowerBound("DEFAULT", "DEFAULT"),
        model.CreateScenarioUpperBound("DEFAULT", "DEFAULT"),
        model.InitialBelief(nullptr));
    despot::ValuedAction result = solver.Search();

    return SearchResult{
      macro_actions[static_cast<size_t>(result.action)],
      static_cast<size_t>(solver.statistics_.num_expanded_nodes),
      static_cast<size_t>(solver.statistics_.longest_trial_length),
      static_cast<float>(result.value)
    };
  }

private:

  class DespotState;
  class DespotBelief;
  class DespotModel;

  class DespotState : public despot::State {
  public:
    T simulation;
    DespotState() { }
  };

  class DespotBelief : public despot::Belief {
  public:

    DespotBelief(const DespotModel* model) : despot::Belief(model) { }

    list_t<despot::State*> Sample(int num) const override {
      list_t<despot::State*> particles;
      for (int i = 0; i < num; i++) {
        particles.push_back(model_->Allocate(i, 1.0 / num));
        if constexpr (!local) {
          static_cast<DespotState*>(particles.back())->simulation =
            static_cast<const DespotModel*>(model_)->GetBelief()->Sample();
        } else {
          static_cast<DespotState*>(particles.back())->simulation =
            static_cast<const DespotModel*>(model_)->GetBelief()->Sample().CreateLocal();
        }
      }
      return particles;
    };

    void Update(despot::ACT_TYPE action, despot::OBS_TYPE obs) override { throw std::logic_error("Unimplemented function."); }
    DespotBelief* MakeCopy() const override { throw std::logic_error("Unimplemented function."); }

  };

  class DespotUpperBound : public despot::ScenarioUpperBound {
  public:
    double Value(const list_t<despot::State*>& particles,
      despot::RandomStreams& streams, despot::History& history) const override {
      double value = 0;
      for (despot::State* particle : particles) {
        value += particle->weight * static_cast<double>(static_cast<DespotState*>(particle)->simulation.BestReward());
      }
      return value;
    }
  };

  // IMPORTANT:
  // We always return 0, since we configure DespotDefaultPolicy to always perform a full
  // roll-out to the end of the planning depth. Thus, when stream is exhausted, i.e. the
  // max planning depth has been reached, we return 0 since the entire planing horizon
  // has been accounted for (we are ignoring everything beyond that).
  class DespotTrivialParticleLowerBound : public despot::ParticleLowerBound {
  public:
    DespotTrivialParticleLowerBound(const despot::DSPOMDP* model)
        : despot::ParticleLowerBound(model) {
    }

    despot::ValuedAction Value(const list_t<despot::State*>& particles) const override {
      despot::ACT_TYPE action = std::uniform_int_distribution<despot::ACT_TYPE>(
          0, static_cast<despot::ACT_TYPE>(model_->NumActions() - 1))(Rng());
      return despot::ValuedAction(action, 0);
    }
  };

  class DespotDefaultPolicy : public despot::DefaultPolicy {
  public:
    DespotDefaultPolicy(const despot::DSPOMDP* model)
        : despot::DefaultPolicy(model, new DespotTrivialParticleLowerBound(model)) { }

    despot::ACT_TYPE Action(const list_t<despot::State*>& particles,
        despot::RandomStreams& streams, despot::History& history) const {
      return std::uniform_int_distribution<despot::ACT_TYPE>(
          0, static_cast<despot::ACT_TYPE>(model_->NumActions() - 1))(Rng());
    }
  };

  class DespotModel : public despot::DSPOMDP {
  public:

    DespotModel(const TBelief* belief, const list_t<list_t<typename T::Action>> actions)
        : _belief(belief), _actions(actions) {

    }

    const TBelief* GetBelief() const { return _belief; }

    // Returns total number of actions.
    int NumActions() const override { return static_cast<int>(_actions.size()); }

    // Deterministic simulative model.
    bool Step(despot::State& state, double random_num, despot::ACT_TYPE action,
        double& reward, despot::OBS_TYPE& obs) const override {

      if (static_cast<DespotState&>(state).simulation.IsTerminal()) {
        throw std::logic_error("Stepping a terminal state.");
      }

      const T& simulation = static_cast<DespotState&>(state).simulation;
      RngDet(true, random_num); // Determinize.
      ExecutionResult execution_result = ExecuteMacroAction(simulation, _actions[static_cast<size_t>(action)]);
      RngDet(true, std::uniform_real_distribution<double>(0.0, 1.0)(Rng())); // Undeterminize.
      reward = execution_result.reward;


      uint64_t obs_hash = 0;
      for (const typename T::Observation& obs : execution_result.observation_trajectory) {
        boost::hash_combine(obs_hash, obs.Discretize());
      }
      obs = obs_hash;

      //obs = execution_result.observation_trajectory.back().Discretize();
      static_cast<DespotState&>(state).simulation = execution_result.state_trajectory.back();
      if (static_cast<DespotState&>(state).simulation.IsTerminal()) {
        return true;
      } else {
        return false;
      }
    }

    // Functions related to beliefs and starting states.
    double ObsProb(despot::OBS_TYPE obs, const despot::State& state, despot::ACT_TYPE action) const override {
      throw std::logic_error("Unimplemented function.");
    }

    despot::Belief* InitialBelief(const despot::State* start, std::string type = "DEFAULT") const override {
      return new DespotBelief(this);
    }

    // Bound-related functions.
    double GetMaxReward() const override { throw std::logic_error("Unimplemente function."); }
    despot::ValuedAction GetBestAction() const override { throw std::logic_error("Unimplemented function."); }

    despot::ScenarioUpperBound* CreateScenarioUpperBound(std::string name="DEFAULT",
        std::string particle_bound_name="DEFAULT") const override {
      return new DespotUpperBound();
    }

    despot::ScenarioLowerBound* CreateScenarioLowerBound(std::string name="DEFAULT",
        std::string particle_bound_name="DEFAULT") const override {
      return new DespotDefaultPolicy(this);
    }

     // Memory management.
    despot::State* Allocate(int state_id, double weight) const override {
      DespotState* state = _memory_pool.Allocate();
      state->state_id = state_id;
      state->weight = weight;
      return state;
    }

    despot::State* Copy(const despot::State* particle) const override {
      DespotState* state = _memory_pool.Allocate();
      *state = *static_cast<const DespotState*>(particle);
      return state;
    }

    void Free(despot::State* particle) const override {
      _memory_pool.Free(static_cast<DespotState*>(particle));
    }

    int NumActiveParticles() const override {
      return static_cast<int>(_memory_pool.num_allocated());
    }

    // Display Functions.
    void PrintState(const despot::State& state, std::ostream& out = std::cout) const override { }
    void PrintObs(const despot::State& state, despot::OBS_TYPE obs, std::ostream& out = std::cout) const override { }
    void PrintAction(despot::ACT_TYPE action, std::ostream& out = std::cout) const override { }
    void PrintBelief(const despot::Belief& belief, std::ostream& out = std::cout) const override { }

  private:
    const TBelief* _belief;
    mutable despot::MemoryPool<DespotState> _memory_pool;
    list_t<list_t<typename T::Action>> _actions;

  };

};

}
