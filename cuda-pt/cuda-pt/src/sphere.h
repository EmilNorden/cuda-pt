#ifndef SPHERE_H_
#define SPHERE_H_

#include "vector3.h"
#include "error_assertion.h"
#include "object.h"
#include "ray.h"


class Sphere : public Object
{
public:
	double radius;
	Vector3d position;
	Vector3d emissive;
	Vector3d diffuse;
	double refl_coeff;

	CUDA_CALLABLE Sphere()
	{
	}

	CUDA_CALLABLE Sphere(const Sphere& other)
	{
		radius = other.radius;
		position = other.position;
		emissive = other.emissive;
		diffuse = other.diffuse;
		refl_coeff = other.refl_coeff;
	}

	__device__ bool intersect(const Ray &ray, IntersectionInfo  &ii) const
	{
		/*double num1 = position.x() - ray.origin_.x();
		double num2 = position.y() - ray.origin_.y();
		double num3 = position.z() - ray.origin_.z();
		double num4 = (num1 * num1) + (num2 * num2) + (num3 * num3);
		double num5 = (radius * radius);
	
		double num6 = (num1 * ray.direction_.x()) 
						+ (num2 * ray.direction_.y()) + (num3 * ray.direction_.z());

		if (num6 < 0.0)
		return false;
		double num7 = num4 - num6 * num6;
		if (num7 > num5)
			return false;

		ii.distance = abs(num6 - sqrt(num5 - num7));
		ii.coordinate = ray.origin_ + (ray.direction_ * ii.distance);
		ii.surface_normal = ii.coordinate - position;

		ii.surface_normal.normalize_device();*/

		return true;
	}
};

#endif
