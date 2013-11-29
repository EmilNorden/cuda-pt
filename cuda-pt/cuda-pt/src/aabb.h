#ifndef AABB_H_
#define AABB_H_

#include "vector3.h"
#include "ray.h"

class AABB
{
private:
	Vector3d min_;
	Vector3d max_;
public:
	CUDA_CALLABLE AABB();
	CUDA_CALLABLE AABB(const Vector3d &min, const Vector3d &max);

	CUDA_CALLABLE Vector3d minv() const { return min_; }
	CUDA_CALLABLE Vector3d maxv() const { return max_; }

	CUDA_CALLABLE void set_min(const Vector3d &min) { min_ = min; }
	CUDA_CALLABLE void set_max(const Vector3d &max) { max_ = max; }

	CUDA_CALLABLE void inflate(const AABB &other);
	CUDA_CALLABLE bool intersects(const AABB &other) const;

	/* Algorithm based on article: http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm */
	CUDA_CALLABLE bool intersects(const Ray &ray) const;
};

#endif