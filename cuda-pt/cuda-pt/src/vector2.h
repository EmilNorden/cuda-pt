#ifndef VECTOR2_H_
#define VECTOR2_H_

#include "error_assertion.h"

template <class T>
class Vector2
{
private:
	T v_[2];
public:
	CUDA_CALLABLE Vector2() {
		v_[0] = T();
		v_[1] = T();
	}

	CUDA_CALLABLE Vector2(T x, T y) {
		v_[0] = x;
		v_[1] = y;
	}

	CUDA_CALLABLE T& x() {
		return v_[0];
	}

	CUDA_CALLABLE T& y() {
		return v_[1];
	}

	CUDA_CALLABLE T x() const {
		return v_[0];
	}

	CUDA_CALLABLE T y() const {
		return v_[1];
	}

	CUDA_CALLABLE bool operator!=(const Vector2 &other) const {
		return v_[0] != other.v_[0] || v_[1] != other.v_[1] || v_[2] != other.v_[2];
	}

	Vector2 operator-(const Vector2 &other) const {
		return Vector2(v_[0] - other.v_[0], v_[1] - other.v_[1]);
	}

	Vector2 operator+(const Vector2 &other) const {
		return Vector2(v_[0] + other.v_[0], v_[1] + other.v_[1]);
	}

	Vector2 operator*(const double f) const {
		return Vector2(v_[0] * f, v_[1] * f);
	}

	double length() const {
		return sqrt(pow(v_[0], 2) + pow(v_[1], 2));
	}

	void normalize() {
		double len = length();
		v_[0] /= len;
		v_[1] /= len;
	}
};

typedef Vector2<int> Vector2i;
typedef Vector2<double> Vector2d;

#endif