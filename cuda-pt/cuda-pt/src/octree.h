#ifndef OCTREE_H_
#define OCTREE_H

#include "aabb.h"
#include "device_list.h"
#include "error_assertion.h"
#include "sphere.h"

struct Cuboid
{
	DeviceList<Sphere> spheres;
	Cuboid *children;
	AABB bounds;
};

class Octree
{
private:
	Cuboid root_;

	CUDA_CALLABLE void build_internal(Cuboid &current, int depth);
public:
	DeviceList<Sphere> spheres;

	CUDA_CALLABLE void build();
};

#endif