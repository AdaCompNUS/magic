#include "core/BezierCurve.h"

#include "core/Util.h"
#include <boost/math/quadrature/trapezoidal.hpp>

BezierCurve::BezierCurve(const vector_t& control0, const vector_t& control1, const vector_t& control2, const vector_t& control3)
    : _control0(control0), _control1(control1), _control2(control2), _control3(control3) {

}

vector_t BezierCurve::Position(float t) const {
  if (t <= 0) {
    return _control0;
  } else if (t >= 1) {
    return _control3;
  } else {
    return (1 - t) * (1 - t) * (1 - t) * _control0 +
      3 * (1 - t) * (1 - t) * t * _control1 +
      3 * (1 - t) * t * t * _control2 +
      t * t * t * _control3;
  }
}

vector_t BezierCurve::Heading(float t) const {
  if (t <= 0) {
    return (_control1 - _control0).normalized();
  } else if (t >= 1) {
    return (_control3 - _control2).normalized();
  } else {
    return (3 * (1 - t) * (1 - t) * (_control1 - _control0) +
      6 * (1 - t) * t * (_control2 - _control1) +
      3 * t * t * (_control3 - _control2)).normalized();
  }
}

float BezierCurve::Length(float start, float end) const {
  if (start >= end) {
    return 0;
  } else {
    return boost::math::quadrature::trapezoidal(
        [this](float t) {
          vector_t grad = -3 * (1 - t) * (1 - t) * _control0 +
              3 * (1 - t) * (1 - 3 * t) * _control1 +
              3 * t * (2 - 3 * t) * _control2 +
              3 * t * t * _control3;
          return grad.norm();
        },
        std::max(start, 0.0f), std::min(end, 1.0f), 1e-12f);
  }
}
