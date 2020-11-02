#include "core/Util.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/device/null.hpp"

RVO::Vector2 ToRVO(const vector_t& v) {
  return RVO::Vector2(static_cast<float>(v.x), static_cast<float>(v.y));
}

vector_t FromRVO(const RVO::Vector2& v) {
  return vector_t(v.x(), v.y());
}

float AngleTo(const vector_t& from, const vector_t& to) {
  return RectifyAngle(atan2f(from.x * to.y - from.y * to.x, from.x * to.x + from.y * to.y));
}

vector_t ClipSpeed(const vector_t& velocity, float speed) {
  if (velocity.norm() > speed) {
    return speed * velocity.normalized();
  } else {
    return velocity;
  }
}

float RectifyAngle(float angle) {
  return angle - (ceilf((angle + PI)/(2 * PI)) - 1) * 2 * PI;
}

float NormalLogProb(float mean, float std, float x) {
  static constexpr float c = -0.9189385332046727f; // -0.5 log(2pi)
  return c - logf(std) - (x - mean) * (x - mean) / (2 * std * std);
}

rng_t& Rng() {
  static thread_local rng_t rng(std::random_device{}());
  return rng;
}

rng_t& RngDet(bool seed, double seed_value) {
  static thread_local rng_t rng(std::random_device{}());
  if (seed) {
    rng = rng_t(static_cast<uint64_t>(seed_value * static_cast<double>(std::numeric_limits<uint64_t>::max())));
  }
  return rng;
}

list_t<float> SoftMax(const list_t<float>& logits) {
  list_t<float> probabilities(logits.size());

  float logit_max = *std::max_element(logits.begin(), logits.end());
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
