#include "experiments/Config.h"
#include "core/BezierCurve.h"
#include <iostream>
#include <macaron/Base64.h>
#include <opencv2/imgcodecs.hpp>

std::string get_cin_line() {
  std::string line;
  std::getline(std::cin, line);
  return line;
}

std::string SerializeBelief(const ExpBelief& belief) {
  list_t<float> values;
  for (size_t i = 0; i < BELIEF_SAMPLES; i++) {
    belief.Sample().Encode(values);
  }
  return macaron::Base64::Encode(ToBytes(values));
}

std::string SerializeFrame(const cv::Mat& frame) {
  std::vector<uchar> buffer;
  std::vector<int> params(2);
  params[0] = cv::IMWRITE_JPEG_QUALITY;
  params[1] = 95;
  cv::imencode(".jpg", frame, buffer, params);
  return macaron::Base64::Encode(ToBytes(buffer));
}

#ifndef SIM_VdpTag
list_t<vector_t> CurveToMacroAction(const BezierCurve& curve, float action_length, size_t depth) {
  list_t<vector_t> macro_action;
  float t = 0.0f;
  for (size_t i = 0; i < depth; i++) {
    float lookahead = t;
    while (lookahead < 1.0 && (curve.Position(lookahead + 1e-5f) - curve.Position(t)).norm() < action_length) {
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

list_t<vector_t> StretchedCurveToMacroAction(const BezierCurve& curve, float action_length, size_t depth) {
  float stretch_lower = 1.0f;
  float stretch_upper = 2.0f;
  bool lower_done = false;
  bool upper_done = false;

  auto get_macro_action = [&](float stretch) {
    return CurveToMacroAction(BezierCurve(
          curve.Control0() * stretch,
          curve.Control1() * stretch,
          curve.Control2() * stretch,
          curve.Control3() * stretch),
        action_length, depth);
  };

  while (true) {
    if (!upper_done) {
      if (get_macro_action(stretch_upper).size() >= depth) {
        upper_done = true;
      } else {
        stretch_upper *= 2.0f;
      }
    } else if (!lower_done) {
      if (get_macro_action(stretch_lower).size() < depth) {
        lower_done = true;
      } else {
        stretch_lower *= 0.5f;
      }
    } else if (stretch_upper - stretch_lower > 1e-4f) {
      float stretch_mid = (stretch_lower + stretch_upper) / 2;
      if (get_macro_action(stretch_mid).size() >= depth) {
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

list_t<list_t<ExpSimulation::Action>> DeserializeMacroActions(const std::string& str, size_t macro_length) {

  list_t<float> macro_curve_params = FromBytes<float>(macaron::Base64::Decode(str));

  // Extract macro-curves.
  list_t<BezierCurve> macro_curves;
  for (size_t i = 0; i * 6 < macro_curve_params.size(); i++) {
    macro_curves.emplace_back(
          vector_t(0.0f, 0.0f),
          vector_t(macro_curve_params[i * 6 + 0], macro_curve_params[i * 6 + 1]),
          vector_t(macro_curve_params[i * 6 + 2], macro_curve_params[i * 6 + 3]),
          vector_t(macro_curve_params[i * 6 + 4], macro_curve_params[i * 6 + 5]));
  }

  // Extract macro-actions from macro-curves.
  list_t<list_t<ExpSimulation::Action>> macro_actions;
  for (const BezierCurve& curve : macro_curves) {
    list_t<vector_t> macro_action = StretchedCurveToMacroAction(
          curve, 1, macro_length);
    macro_actions.emplace_back();
    for (const vector_t& a : macro_action) {
      macro_actions.back().push_back({atan2f(a.y, a.x)});
    }
  }

  return macro_actions;
}
#endif
