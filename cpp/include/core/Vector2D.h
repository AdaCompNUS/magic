/*
Copyright (c) 2020 Chan Jer Shyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <carla/geom/Vector2D.h>
#include <carla/gamma/Vector2.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <ostream>
#include <vector>

class Vector2D {
public:
	float x, y;

	constexpr Vector2D() :x(0), y(0) {}
	constexpr Vector2D(float x, float y) : x(x), y(y) {}
	constexpr Vector2D(const Vector2D& v) : x(v.x), y(v.y) {}
  constexpr Vector2D(const carla::geom::Vector2D& v) : x(v.x), y(v.y) {}
  Vector2D(const carla::gamma::Vector2& v) : x(v.x()), y(v.y()) {}

  operator carla::geom::Vector2D() const { return {x, y}; }
  operator carla::gamma::Vector2() const { return {x, y}; }

  constexpr bool operator==(const Vector2D& v) const {
    return x == v.x && y == v.y;
  }

  constexpr bool operator!=(const Vector2D& v) const {
    return !(x == y);
  }

	constexpr Vector2D& operator=(const Vector2D& v) {
		x = v.x;
		y = v.y;
		return *this;
	}

  constexpr Vector2D operator-() const {
    return Vector2D(-x, -y);
  }

	constexpr Vector2D operator+(const Vector2D& v) const {
		return Vector2D(x + v.x, y + v.y);
	}

	constexpr Vector2D operator-(const Vector2D& v) const {
		return Vector2D(x - v.x, y - v.y);
	}

	constexpr Vector2D& operator+=(const Vector2D& v) {
		x += v.x;
		y += v.y;
		return *this;
	}

  constexpr Vector2D& operator-=(const Vector2D& v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}

	constexpr Vector2D operator*(float s) const {
		return Vector2D(x * s, y * s);
	}

	constexpr Vector2D operator/(float s) const {
		return Vector2D(x / s, y / s);
	}

	constexpr Vector2D& operator*=(float s) {
		x *= s;
		y *= s;
		return *this;
	}

	constexpr Vector2D& operator/=(float s) {
		x /= s;
		y /= s;
		return *this;
	}

	inline void rotate(float theta) {
		float c = cosf(theta);
		float s = sinf(theta);
    float _x = x;
    float _y = y;
		x = _x * c - _y * s;
		y = _x * s + _y * c;
	}

  constexpr Vector2D rotated(float theta) const {
		float c = cosf(theta);
		float s = sinf(theta);
		return Vector2D(x * c - y * s, x * s + y * c);
  }

  constexpr float norm() const {
		return sqrtf(x * x + y * y);
  }

  constexpr float squaredNorm() const {
    return x * x + y * y;
  }

	constexpr void normalize() {
    float n = norm();
		if (n > 0) {
      *this *= 1.0f / n;
    }
	}

  constexpr Vector2D normalized() const {
    float n = norm();
		if (n == 0) return *this;
    return *this * (1.0f / n);
  }

  constexpr float dot(const Vector2D& v) const {
    return v.x * x + v.y * y;
  }

  inline void Encode(std::vector<float>& data) const {
    data.emplace_back(x);
    data.emplace_back(y);
  }

  static inline Vector2D Decode(std::vector<float>::const_iterator& data) {
    Vector2D v;
    v.x = *data; data++;
    v.y = *data; data++;
    return v;
  }
};

constexpr inline Vector2D operator*(float s, const Vector2D& v) {
  return Vector2D(s * v.x, s * v.y);
}

inline std::ostream& operator<<(std::ostream& os, const Vector2D& v)
{
  os << "(" << v.x << ", " << v.y << ")";
  return os;
}

BOOST_GEOMETRY_REGISTER_POINT_2D(Vector2D, float, cs::cartesian, x, y)
