#pragma once

#include "core/Types.h"
#include <iostream>

inline float RelaxedDubinsLength(vector_t p, float r)
{
  if (p.y < 0) p.y *= -1;

  float r2 = r * r;
  vector_t p_rel(p.x, p.y - r);
  float p_rel_norm2 = p_rel.squaredNorm();

  if (p_rel_norm2 >= r2)
  {
    float a = PI - atan2f(p_rel.x, p_rel.y) - acosf(r / sqrtf(p_rel_norm2));
    return r * a + sqrtf(p_rel_norm2 + r2);
  }
  else
  {
    float g_c1_2 = p.x * p.x + (p.y - r) * (p.y - r);
    float g_c2_2 = p.x * p.x + (p.y + r) * (p.y + r);
    float g_c2 = sqrtf(g_c2_2);
    float c1_c2 = 2 * r;
    float c1_c2_2 = c1_c2 * c1_c2;
    float a1 = acosf((g_c2_2 + c1_c2_2 - g_c1_2) / (2 * g_c2 * c1_c2));
    float a2 = acosf((c1_c2_2 + g_c2_2 - r * r) / (2 * c1_c2 * g_c2));
    float b = 2 * PI - acosf(((g_c2_2 / (r * r)) - 5) / (-4));
    return r * (a1 + a2 + b);
  }
}

inline float RelaxedDubinsLength(const vector_t& start, const vector_t& heading, const vector_t& end, float r) {
  vector_t relative_position = end - start;
  relative_position.rotate(AngleTo(heading, vector_t(1, 0)));
  return RelaxedDubinsLength(relative_position, r);
}
