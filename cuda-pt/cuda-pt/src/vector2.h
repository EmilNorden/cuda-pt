#ifndef VECTOR2_H_
#define VECTOR2_H_

template <class T>
class Vector2
{
private:
	T v_[2];
public:
	Vector2() {
		v_[0] = T();
		v_[1] = T();
	}

	Vector2(T x, T y) {
		v_[0] = x;
		v_[1] = y;
	}

	T& x() {
		return v_[0];
	}

	T& y() {
		return v_[1];
	}

	T x() const {
		return v_[0];
	}

	T y() const {
		return v_[1];
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