#ifndef VECTOR3_H_
#define VECTOR3_H_

#include "curand_kernel.h"
#include "error_assertion.h"
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>

template <class T>
class Vector3
{
private:
	T v_[3];
public:
	CUDA_CALLABLE Vector3() {
		v_[0] = T();
		v_[1] = T();
		v_[2] = T();
	}

	CUDA_CALLABLE Vector3(T x, T y, T z) {
		v_[0] = x;
		v_[1] = y;
		v_[2] = z;
	}

	CUDA_CALLABLE Vector3(const Vector3 &vector) {
		v_[0] = vector.v_[0];
		v_[1] = vector.v_[1];
		v_[2] = vector.v_[2];
	}
	

	CUDA_CALLABLE T& x() {
		return v_[0];
	}

	CUDA_CALLABLE T& y() {
		return v_[1];
	}

	CUDA_CALLABLE T& z() {
		return v_[2];
	}

	CUDA_CALLABLE T x() const {
		return v_[0];
	}

	CUDA_CALLABLE T y() const {
		return v_[1];
	}

	CUDA_CALLABLE T z() const {
		return v_[2];
	}

	CUDA_CALLABLE Vector3 &operator=(const Vector3 &other) {
		v_[0] = other.v_[0];
		v_[1] = other.v_[1];
		v_[2] = other.v_[2];

		return *this;
	}

	CUDA_CALLABLE bool operator==(const Vector3 &other) const {
		return v_[0] == other.v_[0] && v_[1] == other.v_[1] && v_[2] == other.v_[2];
	}

	CUDA_CALLABLE bool operator!=(const Vector3 &other) const {
		return !(*this == other);
	}

	CUDA_CALLABLE Vector3 operator-(const Vector3 &other) const {
		return Vector3(v_[0] - other.v_[0], v_[1] - other.v_[1], v_[2] - other.v_[2]);
	}

	CUDA_CALLABLE Vector3 operator+(const Vector3 &other) const {
		return Vector3(v_[0] + other.v_[0], v_[1] + other.v_[1], v_[2] + other.v_[2]);
	}

	CUDA_CALLABLE Vector3 operator*(const Vector3 &other) const {
		return Vector3(v_[0] * other.v_[0], v_[1] * other.v_[1], v_[2] * other.v_[2]);
	}

	CUDA_CALLABLE Vector3 operator*(const double f) const {
		return Vector3(v_[0] * f, v_[1] * f, v_[2] * f);
	}

	CUDA_CALLABLE Vector3 operator/(const double f) const {
		return Vector3(v_[0] / f, v_[1] / f, v_[2] / f);
	}

	CUDA_CALLABLE Vector3 operator-(const double f) const {
		return Vector3(v_[0] - f, v_[1] - f, v_[2] - f);
	}

	CUDA_CALLABLE Vector3 operator+(const double f) const {
		return Vector3(v_[0] + f, v_[1] + f, v_[2] + f);
	}

	CUDA_CALLABLE Vector3 &operator+=(const Vector3 &other) {
		v_[0] += other.v_[0];
		v_[1] += other.v_[1];
		v_[2] += other.v_[2];

		return *this;
	}

	CUDA_CALLABLE double length() const {
		return sqrt(pow(v_[0], 2) + pow(v_[1], 2) + pow(v_[2], 2));
	}

	CUDA_CALLABLE double length_squared() const {
		return pow(v_[0], 2) + pow(v_[1], 2) + pow(v_[2], 2);
	}

	CUDA_CALLABLE void normalize() {
		double len = length();
		v_[0] /= len;
		v_[1] /= len;
		v_[2] /= len;
	}

	CUDA_CALLABLE Vector3 normal_component(const Vector3 &n) const {
		return n * (dot(n));
	}

	CUDA_CALLABLE Vector3 tangential_component(const Vector3 &n, const Vector3 &normal_component) const {
		return *this - normal_component;
	}

	CUDA_CALLABLE Vector3 cross(const Vector3 &other) const {
		return Vector3(y() * other.z() - z() * other.y(),
			z() * other.x() - x() * other.z(),
			x() * other.y() - y() * other.x());
	}

	CUDA_CALLABLE double dot(const Vector3 &other) const {
		return x() * other.x() + y() * other.y() + z() * other.z();
	}

	CUDA_CALLABLE bool is_zero() const {
		return !v_[0] && !v_[1] && !v_[2];
	}

	CUDA_CALLABLE void clamp(const Vector3 &min, const Vector3 &max) {
		x() = min.x() < x() ? x() : min.x();
		y() = min.y() < y() ? y() : min.y();
		z() = min.z() < z() ? z() : min.z();

		x() = max.x() > x() ? x() : max.x();
		y() = max.y() > y() ? y() : max.y();
		z() = max.z() > z() ? z() : max.z();
	}

	CUDA_CALLABLE void max_vector(const Vector3 &other)
	{
		if(other.x() > x())
			x() = other.x();
		if(other.y() > y())
			y() = other.y();
		if(other.z() > z())
			z() = other.z();
	}

	CUDA_CALLABLE void min_vector(const Vector3 &other)
	{
		if(other.x() < x())
			x() = other.x();
		if(other.y() < y())
			y() = other.y();
		if(other.z() < z())
			z() = other.z();
	}

	CUDA_CALLABLE void multiply(double factor) {
		v_[0] *= factor;
		v_[1] *= factor;
		v_[2] *= factor;
	}

	CUDA_CALLABLE void flip() {
		v_[0] = -v_[0];
		v_[1] = -v_[1];
		v_[2] = -v_[2];
	}

	__device__ static Vector3 rand_unit_in_hemisphere(const Vector3 &normal, curandState &state)
	{

		//double mt_x = ((rand() / (double)RAND_MAX) * 2) - 1.0;
		//double mt_y = ((rand() / (double)RAND_MAX) * 2) - 1.0;
		//double mt_z = ((rand() / (double)RAND_MAX) * 2) - 1.0;
		
		double mt_x = (curand_uniform(&state) * 2) - 1.0;
		double mt_y = (curand_uniform(&state) * 2) - 1.0;
		double mt_z = (curand_uniform(&state) * 2) - 1.0;

		Vector3 vector(mt_x, mt_y, mt_z);
		vector.normalize();

		if(vector.dot(normal) < 0)
		{
			vector.x() = -vector.x();
			vector.y() = -vector.y();
			vector.z() = -vector.z();
		}

		return vector;
	}
};

typedef Vector3<double> Vector3d;

#endif