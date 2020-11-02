#pragma once

#include "core/Types.h"

class BezierCurve {

public:

  BezierCurve(const vector_t& control0, const vector_t& control1, const vector_t& control2, const vector_t& control3);

  const vector_t& Control0() const { return _control0; }
  const vector_t& Control1() const { return _control1; }
  const vector_t& Control2() const { return _control2; }
  const vector_t& Control3() const { return _control3; }

  vector_t Position(float t) const;
  vector_t Heading(float t) const;
  float Length(float start=0, float end=1) const;

private:

  vector_t _control0;
  vector_t _control1;
  vector_t _control2;
  vector_t _control3;

};
