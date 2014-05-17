#ifndef OBJECT_H_
#define OBJECT_H_

#include "error_assertion.h"
//#include "intersectioninfo.h"

class Ray;
class IntersectionInfo;

class Object
{
	__device__ virtual bool intersect(const Ray &ray, IntersectionInfo  &ii) const = 0;
};

#endif