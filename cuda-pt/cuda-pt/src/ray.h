#ifndef RAY_H_
#define RAY_H_

#include "cuda_helpers.h"
#include "vector3.h"
#include "intersectioninfo.h"

class Ray
{
public:
	CUDA_CALLABLE Ray(int depth = 0) : depth_(depth), reflected(nullptr) {
	}
	int depth_;
	Vector3d origin_;
	Vector3d direction_;
	double refractive_index;
	IntersectionInfo intersection;

	Ray *reflected;
};

#endif