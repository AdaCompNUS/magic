#pragma once

#include "core/BezierCurve.h"
#include "core/Types.h"
#include "rvo2/RVO.h"
#include <boost/functional/hash.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <random>
#include <carla/occupancy/OccupancyMap.h>
#include <iostream>

RVO::Vector2 ToRVO(const vector_t& v);

vector_t FromRVO(const RVO::Vector2& v);

// Signed angle from first vector to second vector. Rotating first
// vector by returned value gives a vector parallel to the second vector.
// Rectified into [-PI, PI].
float AngleTo(const vector_t& from, const vector_t& to);

vector_t ClipSpeed(const vector_t& velocity, float speed);

template <typename T>
T Clamp(const T& value, const T& min, const T& max) {
  if (value <= min) {
    return min;
  } else if (value >= max) {
    return max;
  } else {
    return value;
  }
}

float RectifyAngle(float angle);

float NormalLogProb(float mean, float std, float x);

list_t<float> SoftMax(const list_t<float>& logits);

template <size_t N>
array_t<float, N> SoftMax(const array_t<float, N>& logits) {
  array_t<float, N> probabilities;

  float logit_max = *boost::max_element(logits.begin(), logits.end());
  float normalizing = 0;
  for (size_t i = 0; i < logits.size(); i++) {
    probabilities[i] = expf(logits[i] - logit_max);
    normalizing += probabilities[i];
  }
  for (size_t i = 0; i < logits.size(); i++) {
    probabilities[i] /= normalizing;
  }

  return probabilities;
}

rng_t& Rng();

rng_t& RngDet(bool seed=false, double seed_value=0);

template <typename T>
struct list_hash_t {
    std::size_t operator()(const T& l) const {
        return boost::hash_range(l.begin(), l.end());
    }
};

template <typename T>
std::string ToBytes(const list_t<T>& list) {
  list_t<char> bytes;
  for (const T& i : list) {
    const char* i_bytes = reinterpret_cast<const char*>(&i);
    bytes.insert(bytes.end(), i_bytes, i_bytes + sizeof(T));
  }
  return std::string(bytes.begin(), bytes.end());
}

template <typename T>
list_t<T> FromBytes(const std::string& data) {
  list_t<T> list;
  for (size_t i = 0; i < data.size(); i += sizeof(T)) {
    list.emplace_back(*reinterpret_cast<const T*>(&data[i]));
  }
  return list;
}

template <typename T>
std::optional<T> FindFirstRootQuadratic(const T& a, const T& b, const T& c, const T& min, const T& max) {
  T det = b * b - 4 * a * c;
  if (det < 0) {
    return {};
  } else if (det == 0) {
    T root = -b / (2 * a);
    if (root >= min && root <= max) {
      return root;
    } else {
      return {};
    }
  } else {
    T sqrt_det = sqrtf(det);
    T root = (-b - sqrt_det) / (2 * a);
    if (root >= min && root <= max) {
      return root;
    }
    root = (-b + sqrt_det) / (2 * a);
    if (root >= min && root <= max) {
      return root;
    } else {
      return {};
    }

  }

}

// Discretize curve into macro-action of up to <count> line segments.
list_t<vector_t> StandardCurveDiscretization(const BezierCurve& curve, float action_length, size_t count);

// Discretize curve into macro-action of exactly <count> line segments, with stretching.
list_t<vector_t> StandardStretchedCurveDiscretization(const BezierCurve& curve, float action_length, size_t count);

// Converts macro-action parameters into macro-action set.
template <typename T>
list_t<list_t<typename T::Action>> StandardMacroActionDeserialization(const list_t<float>& params, size_t macro_length) {

  if (params.size() != 6 * 8) {
    throw std::logic_error("Expected 48 macro-action parameters, got " + std::to_string(params.size()));
  }

  // Extract macro-curves.
  list_t<BezierCurve> macro_curves;
  for (size_t i = 0; i * 6 < params.size(); i++) {
    macro_curves.emplace_back(
          vector_t(0.0f, 0.0f),
          vector_t(params[i * 6 + 0], params[i * 6 + 1]),
          vector_t(params[i * 6 + 2], params[i * 6 + 3]),
          vector_t(params[i * 6 + 4], params[i * 6 + 5]));
  }

  // Extract macro-actions from macro-curves.
  list_t<list_t<typename T::Action>> macro_actions;
  for (const BezierCurve& curve : macro_curves) {
    list_t<vector_t> macro_action = StandardStretchedCurveDiscretization(
          curve, 1, macro_length);
    macro_actions.emplace_back();
    for (const vector_t& a : macro_action) {
      macro_actions.back().push_back({atan2f(a.y, a.x)});
    }
  }

  return macro_actions;
}

bool RectangleContains(const array_t<vector_t, 4>& rect, const vector_t& point);

bool RectangleIntersects(const array_t<vector_t, 4>& rect1, const array_t<vector_t, 4>& rect2);

bool ContainsAny(const carla::occupancy::OccupancyMap& area, const array_t<vector_t, 4>& points);

bool ContainsAll(const carla::occupancy::OccupancyMap& area, const array_t<vector_t, 4>& points);
