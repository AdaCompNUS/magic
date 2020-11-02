#pragma once

#include "core/Types.h"
#include "rvo2/RVO.h"
#include <boost/functional/hash.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <random>

RVO::Vector2 ToRVO(const vector_t& v);

vector_t FromRVO(const RVO::Vector2& v);

// Signed angle from first vector to second vector. Rotating first
// vector by returned value gives a vector parallel to the second vector.
// Rectified into [-PI, PI].
float AngleTo(const vector_t& from, const vector_t& to);

vector_t ClipSpeed(const vector_t& velocity, float speed);

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
