#pragma once

#include <cstdint>
#include <limits>

class SplitMix64 {

public:

  typedef uint64_t result_type;
  static constexpr uint64_t min() { return std::numeric_limits<uint64_t>::min(); }
  static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

  SplitMix64(uint64_t seed);

  uint64_t operator()();

private:

  uint64_t _s;

};
