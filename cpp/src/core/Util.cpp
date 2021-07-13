#include "core/Util.h"

#include "boost/iostreams/device/null.hpp"
#include "boost/iostreams/stream.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

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

list_t<vector_t> StandardCurveDiscretization(const BezierCurve& curve, float action_length, size_t count) {
  list_t<vector_t> macro_action;
  float t = 0.0f;
  for (size_t i = 0; i < count; i++) {
    float lookahead = t;
    while (lookahead < 1.0f && (curve.Position(lookahead + 1e-5f) - curve.Position(t)).norm() < action_length) {
      lookahead += 1e-5f;
    }
    if (lookahead >= 1.0f) {
      break;
    }
    macro_action.emplace_back((curve.Position(lookahead) - curve.Position(t)));
    t = lookahead;
  }
  return macro_action;
}

list_t<vector_t> StandardStretchedCurveDiscretization(const BezierCurve& curve, float action_length, size_t count) {
  float stretch_lower = 1.0f;
  float stretch_upper = 2.0f;
  bool lower_done = false;
  bool upper_done = false;

  auto get_macro_action = [&](float stretch) {
    return StandardCurveDiscretization(BezierCurve(
          curve.Control0() * stretch,
          curve.Control1() * stretch,
          curve.Control2() * stretch,
          curve.Control3() * stretch),
        action_length, count);
  };

  while (true) {
    if (!upper_done) {
      if (get_macro_action(stretch_upper).size() >= count) {
        upper_done = true;
      } else {
        stretch_upper *= 2.0f;
      }
    } else if (!lower_done) {
      if (get_macro_action(stretch_lower).size() < count) {
        lower_done = true;
      } else {
        stretch_lower *= 0.5f;
      }
    } else if (stretch_upper - stretch_lower > 1e-4f) {
      float stretch_mid = (stretch_lower + stretch_upper) / 2;
      if (get_macro_action(stretch_mid).size() >= count) {
        stretch_upper = stretch_mid;
      } else {
        stretch_lower = stretch_mid;
      }
    } else {
      break;
    }
  }

  return get_macro_action(stretch_upper);
}

bool RectangleContains(const array_t<vector_t, 4>& rect, const vector_t& point) {
  float v1 = (point - rect[0]).dot(rect[1] - rect[0]);
  if (v1 < 0 || v1 > (rect[1] - rect[0]).squaredNorm()) { // dot(relative vector, dir / norm(dir)) >= norm(dir) iff dot(relative vector, dir) >= norm^2(dir)
    return false;
  }

  float v2 = (point - rect[0]).dot(rect[3] - rect[0]);
  if (v2 < 0 || v2 > (rect[3] - rect[0]).squaredNorm()) {
    return false;
  }

  return true;
}

bool RectangleIntersects(const array_t<vector_t, 4>& rect1, const array_t<vector_t, 4>& rect2) {
  return
      RectangleContains(rect1, rect2[0]) || RectangleContains(rect1, rect2[1]) ||
      RectangleContains(rect1, rect2[2]) || RectangleContains(rect1, rect2[3]) ||
      RectangleContains(rect2, rect1[0]) || RectangleContains(rect2, rect1[1]) ||
      RectangleContains(rect2, rect1[2]) || RectangleContains(rect2, rect1[3]);
}

bool ContainsAny(const carla::occupancy::OccupancyMap& area, const array_t<vector_t, 4>& points) {
  for (const vector_t& p : points) {
    if (area.Contains(p)) {
      return true;
    }
  }
  return false;
}

bool ContainsAll(const carla::occupancy::OccupancyMap& area, const array_t<vector_t, 4>& points) {
  for (const vector_t& p : points) {
    if (!area.Contains(p)) {
      return false;
    }
  }
  return true;
}
