#include "core/random/SplitMix64.h"

SplitMix64::SplitMix64(uint64_t seed) : _s(seed) { }

/* Implementation taken from https://github.com/jj1bdx/xorshiftplus-c/blob/master/splitmix64.c */
uint64_t SplitMix64::operator()() {
	uint64_t z = (_s += static_cast<uint64_t>(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * static_cast<uint64_t>(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * static_cast<uint64_t>(0x94D049BB133111EB);
	return z ^ (z >> 31);
}
