#include "core/random/XorShift128Plus.h"

#include "core/random/SplitMix64.h"
#include <random>

XorShift128Plus::XorShift128Plus(uint64_t seed) {
  SplitMix64 sm(seed);
  _s0 = sm();
  _s1 = sm();
}

XorShift128Plus::XorShift128Plus()
    : XorShift128Plus(std::default_random_engine{}()) {
}

/* Implementation taken from https://github.com/jj1bdx/xorshiftplus-c/blob/master/xorshift128plus.c */
uint64_t XorShift128Plus::operator()() {
	uint64_t s1 = _s0;
	const uint64_t s0 = _s1;
	const uint64_t result = s0 + s1;
	_s0 = s0;
	s1 ^= s1 << 23; // a
	_s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
	return result;
}
