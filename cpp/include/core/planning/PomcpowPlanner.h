#pragma once

#include <ctime>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>

namespace planning {

template <typename T, typename TBelief, bool local=false>
class PomcpowPlanner {

public:

  struct SearchResult {
    typename T::Action action;
    size_t num_nodes;
    size_t depth;
    float value;
  };

  float ucb;
  float k_action;
  float alpha_action;
  float k_observation;
  float alpha_observation;

  PomcpowPlanner(float ucb=T::POMCPOW_UCB,
      float k_action=T::POMCPOW_K_ACTION, float alpha_action=T::POMCPOW_ALPHA_ACTION,
      float k_observation=T::POMCPOW_K_OBSERVATION, float alpha_observation=T::POMCPOW_ALPHA_OBSERVATION)
      : ucb(ucb), k_action(k_action), alpha_action(alpha_action),
      k_observation(k_observation), alpha_observation(alpha_observation) {

  }

  SearchResult Search(const TBelief& belief) const {

    // Initialize node storage.
    list_t<PlannerNode*> storage;
    auto create_node = [&]() {
      storage.emplace_back(new PlannerNode());
      return storage.back();
    };

    // Initialize statistics.
    size_t max_depth = 0;
    size_t num_nodes = 1;

    // Simulate function.
    std::function<float(PlannerNode*, const T&, size_t)> simulate = [&](PlannerNode* node, const T& state, size_t remaining_depth) {

      if (state.IsTerminal()) {
        return 0.0f;
      }

      if (remaining_depth == 0) {
        return 0.0f;
      }

      // Initialize fields.
      float total = 0;

      // Action selection.
      size_t action = SelectAction(node);

      // Check pw limit.
      bool pw_limit_okay = CheckPwLimitObservation(node, action);
      size_t resampled_observation;
      if (!pw_limit_okay) {
        // Line 9: Uniformly sample existing observation.
        resampled_observation = SampleObservation(node, action);
      }

      // Sample observation (sample state, and execute action).
      std::tuple<T, float, typename T::Observation, float> step_result = state.template Step<true>(
          node->actions[action],
          pw_limit_okay ? nullptr : &node->observations[action][resampled_observation]);

      // Add to (action, reward, next_state) lists.
      node->action_rewards[action].emplace_back(std::get<1>(step_result));
      node->action_next_states[action].emplace_back(std::get<0>(step_result));

      // Check progressive widening limit.
      if (pw_limit_okay) {

        // Line 13: Add observation to list.
        node->observations[action].emplace_back(std::get<2>(step_result));

        // Line 10: Add next_state to list under observation.
        node->observation_next_states[action].emplace_back();
        node->observation_next_states[action].back().emplace_back(node->action_next_states[action].size() - 1);

        // Line 11: Add observation probabiliy to list under observation.
        node->observation_probabilities[action].emplace_back();
        node->observation_probabilities[action].back().emplace_back(std::get<3>(step_result));

        // Create child node.
        node->observation_children[action].emplace_back(create_node());
        num_nodes++;

        // Estimate value using rollout.
        total = node->action_rewards[action].back();
        total += T::GAMMA * Rollout(node->action_next_states[action].back(), remaining_depth - 1);
      } else {

        size_t observation = resampled_observation;

        // Line 10: Add next_state to list under observation.
        node->observation_next_states[action][observation].emplace_back(node->action_next_states[action].size() - 1);

        // Line 11: Add observation probability to list under observation.
        node->observation_probabilities[action][observation].emplace_back(std::get<3>(step_result));

        // Line 16: Resmaple next_state
        size_t next_state = SampleObservationNextState(node, action, observation);

        // Line 17: Reuse reward. Line 18: Estimate value recursively.
        total = node->action_rewards[action][next_state];
        total += T::GAMMA * simulate(
            node->observation_children[action][observation],
            node->action_next_states[action][next_state],
            remaining_depth - 1);
      }

      // Update counts and values
      node->visitation_count++;
      node->action_visitation_counts[action]++;
      node->action_values[action] += (total - node->action_values[action]) / static_cast<float>(node->action_visitation_counts[action]);

      // Update max depth.
      max_depth = std::max(max_depth, T::SEARCH_DEPTH - remaining_depth + 1);

      return total;
    };

    // Initialize belief tree.
    PlannerNode* tree = create_node();

    // Initialize duration tracking fields.
    auto start_time_cpu = std::clock();
    size_t num_iterations = 0;
    while (true) {

      // Check if predicted duration exceeds time limit.
      if (num_iterations > 0) {
        size_t elapsed_cpu = static_cast<size_t>(1000 * (std::clock() - start_time_cpu) / CLOCKS_PER_SEC);
        if (elapsed_cpu * (num_iterations + 1) >= T::PLANNING_TIME * num_iterations) {
          break;
        }
      }

      // Invoke simulate function.
      if constexpr (!local) {
        simulate(tree, belief.Sample(), T::SEARCH_DEPTH);
      } else {
        simulate(tree, belief.Sample().CreateLocal(), T::SEARCH_DEPTH);
      }
      num_iterations++;
    }


    float best_value = std::numeric_limits<float>::quiet_NaN();
    typename T::Action best_action;

    if (tree->action_values.size() == 0) {
      best_action = T::Action::Rand();
    } else {
      // Select best action and return.
      size_t best_index = static_cast<size_t>(std::distance(
            tree->action_values.begin(),
            std::max_element(tree->action_values.begin(), tree->action_values.end())));
      best_value = tree->action_values[best_index];
      best_action = tree->actions[best_index];
    }

    // Delete nodes.
    for (PlannerNode* deletion_node : storage) {
      delete deletion_node;
    }

    // Return action and statistics.
    return SearchResult{best_action, num_nodes, max_depth, best_value};
  }

private:

  struct PlannerNode {
    size_t visitation_count;

    // Visitation count and values for each action.
    list_t<typename T::Action> actions;
    std::unordered_set<float> action_ids;
    list_t<size_t> action_visitation_counts;
    list_t<float> action_values;

    // Lists of all (action, reward, next_state) for each action.
    list_t<list_t<float>> action_rewards;
    list_t<list_t<T>> action_next_states;

    // Children after observation branches.
    list_t<list_t<typename T::Observation>> observations; // One list per action.
    list_t<list_t<list_t<size_t>>> observation_next_states; // s' \in B(h, a, o)
    list_t<list_t<list_t<float>>> observation_probabilities; // Z(o | s, z, s')
    list_t<list_t<PlannerNode*>> observation_children;

    PlannerNode() : visitation_count(0) { }
  };

  // Select first unvisited action, if any.
  // Else, select action with best augmented value.
  size_t SelectAction(PlannerNode* node) const {
    if (CheckPwLimitAction(node)) {
      typename T::Action action;
      do {
        action = T::Action::Rand();
      } while (node->action_ids.find(action.Id()) != node->action_ids.end());

      node->actions.emplace_back(action);
      node->action_ids.emplace(action.Id());
      node->action_visitation_counts.emplace_back(0);
      node->action_values.emplace_back(0);

      node->action_rewards.emplace_back();
      node->action_next_states.emplace_back();

      node->observations.emplace_back();
      node->observation_next_states.emplace_back();
      node->observation_probabilities.emplace_back();
      node->observation_children.emplace_back();
    }

    float best_augmented_value = -std::numeric_limits<float>::infinity();
    size_t best_index = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < node->action_visitation_counts.size(); i++) {
      if (node->action_visitation_counts[i] == 0) {
        return i;
      }
      float augmented_value = node->action_values[i];
      augmented_value += this->ucb * sqrtf(
          logf(static_cast<float>(node->visitation_count)) / static_cast<float>(node->action_visitation_counts[i]));
      if (augmented_value > best_augmented_value) {
        best_augmented_value = augmented_value;
        best_index = i;
      }
    }
    return best_index;
  };

  // Sample observation uniformly.
  size_t SampleObservation(const PlannerNode* node, size_t action) const {
    return std::uniform_int_distribution<size_t>(0, node->observations[action].size() - 1)(Rng());
  };

  // Sample state under observation uniformly.
  size_t SampleObservationNextState(const PlannerNode* node, size_t action, size_t observation) const {
    return node->observation_next_states[action][observation][
      std::discrete_distribution<size_t>(
        node->observation_probabilities[action][observation].begin(),
        node->observation_probabilities[action][observation].end())(Rng())];
  };

  // Check observation progressive widening limit.
  bool CheckPwLimitAction(const PlannerNode* node) const {
    return static_cast<float>(node->action_visitation_counts.size()) <=
        this->k_action * powf(static_cast<float>(node->visitation_count), this->alpha_action);
  };

  // Check observation progressive widening limit.
  bool CheckPwLimitObservation(const PlannerNode* node, size_t action) const {
    return static_cast<float>(node->observation_children[action].size()) <=
        this->k_observation * powf(static_cast<float>(node->action_visitation_counts[action]), this->alpha_observation);
  };

  // Random rollout.
  float Rollout(T sim, size_t remaining_depth) const {

    if (sim.IsTerminal()) {
      return 0.0f;
    }

    float reward = 0.0;
    float discount = 1.0;

    while (true) {

      if (remaining_depth == 0) {
        return reward;
      }

      // Step.
      std::tuple<T, float, typename T::Observation, float> step_result = sim.template Step<false>(T::Action::Rand());

      // Update reward, discount, depth.
      reward += discount * std::get<1>(step_result);
      if (std::get<0>(step_result).IsTerminal()) {
        return reward;
      }
      discount *= T::GAMMA;
      remaining_depth--;

      // Update sim.
      sim = std::get<0>(step_result);
    }

  };

};

}
