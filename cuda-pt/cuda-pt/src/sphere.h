#ifndef SPHERE_H_
#define SPHERE_H_

#include "vector3.h"

class Sphere
{
public:
	double radius;
	Vector3d position;
	Vector3d emissive;
	Vector3d diffuse;
	double refl_coeff;
};

#endif