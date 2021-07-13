#pragma once

#include "core/random/XorShift128Plus.h"
#include <array>
#include <cstdint>
#include <math.h>
#include <vector>

template <typename T>
using list_t = std::vector<T>;

template <typename T, size_t N>
using array_t = std::array<T, N>;

typedef XorShift128Plus rng_t;

#include "core/Vector2D.h"
typedef Vector2D vector_t;

static constexpr float PI = static_cast<float>(M_PI);
