#ifndef RAY_H_
#define RAY_H_

#include "cuda_helpers.h"
#include "vector3.h"

class Ray
{
public:
	CUDA_CALLABLE Ray() : depth_(0) {
	}
	int depth_;
	Vector3d origin_;
	Vector3d direction_;
	double refractive_index;

};

#endif