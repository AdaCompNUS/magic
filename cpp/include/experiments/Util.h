#pragma once

#include "core/BezierCurve.h"
#include "experiments/Config.h"
#include <iostream>
#include <macaron/Base64.h>
#include <opencv2/imgcodecs.hpp>

std::string get_cin_line() {
  std::string line;
  std::getline(std::cin, line);
  return line;
}

template <typename T>
std::string SerializeBelief(const T& belief) {
  list_t<float> values;
  for (size_t i = 0; i < BELIEF_SAMPLES; i++) {
    belief.Sample().Encode(values);
  }
  return macaron::Base64::Encode(ToBytes(values));
}

template <typename T>
std::string SerializeContext() {
  list_t<float> values;
  T::EncodeContext(values);
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
