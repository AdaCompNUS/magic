#pragma once

#include <cstdint>
#include <limits>

class XorShift128Plus {

public:

  typedef uint64_t result_type;
  static constexpr uint64_t min() { return std::numeric_limits<uint64_t>::min(); }
  static constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

  XorShift128Plus();
  XorShift128Plus(uint64_t seed);

  uint64_t operator()();

private:

  uint64_t _s0, _s1;

};
