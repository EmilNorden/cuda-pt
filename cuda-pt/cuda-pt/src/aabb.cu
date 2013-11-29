#include "aabb.h"

#define INTERSECT_EPSILON 9.99999997475243E-07

AABB::AABB()
	: min_(0, 0, 0), max_(0, 0, 0)
{
}

AABB::AABB(const Vector3d &min, const Vector3d &max)
	: min_(min), max_(max)
{
}

void AABB::inflate(const AABB &other)
{
	min_.x() = std::min(min_.x(), other.min_.x());
	min_.y() = std::min(min_.y(), other.min_.y());

	max_.x() = std::max(max_.x(), other.max_.x());
	max_.y() = std::max(max_.y(), other.max_.y());
}


bool AABB::intersects(const AABB &other) const
{
	return max_.x() >= other.min_.x() && min_.x() <= other.max_.x() && 
		max_.y() >= other.min_.y() && min_.y() <= other.max_.y();
}

/* Algorithm based on article: http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm */
bool AABB::intersects(const Ray &ray) const
{
	double tNear = DBL_MIN;
	double tFar = DBL_MAX;

	if(abs(ray.direction_.x()) < INTERSECT_EPSILON)
	{
		if(ray.origin_.x() < this->min_.x() || ray.origin_.x() > this->max_.x())
		{
			return false;
		}
	}
	else
	{
		double T1 = (this->min_.x() - ray.origin_.x()) / ray.direction_.x();
		double T2 = (this->max_.x() - ray.origin_.x()) / ray.direction_.x();
		if(T1 > T2)
		{
			double temp = T1;
			T1 = T2;
			T2 = temp;
		}

		if(T1 > tNear)
		{
			tNear = T1;
		}

		if(T2 < tFar)
		{
			tFar = T2;
		}
			

		if(tNear > tFar || tFar < 0)
		{
			return false;
		}
	}

	if(abs(ray.direction_.y()) < INTERSECT_EPSILON)
	{
		if(ray.origin_.y() < this->min_.y() || ray.origin_.y() > this->max_.y())
		{
			return false;
		}
	}
	else
	{
		double T1 = (this->min_.y() - ray.origin_.y()) / ray.direction_.y();
		double T2 = (this->max_.y() - ray.origin_.y()) / ray.direction_.y();
		if(T1 > T2)
		{
			double temp = T1;
			T1 = T2;
			T2 = temp;
		}

		if(T1 > tNear)
		{
			tNear = T1;
		}

		if(T2 < tFar)
		{
			tFar = T2;
		}
			

		if(tNear > tFar || tFar < 0)
		{
			return false;
		}
	}


	if(abs(ray.direction_.z()) < INTERSECT_EPSILON)
	{
		if(ray.origin_.z() < this->min_.z() || ray.origin_.z() > this->max_.z())
		{
			return false;
		}
	}
	else
	{
		double T1 = (this->min_.z() - ray.origin_.z()) / ray.direction_.z();
		double T2 = (this->max_.z() - ray.origin_.z()) / ray.direction_.z();
		if(T1 > T2)
		{
			double temp = T1;
			T1 = T2;
			T2 = temp;
		}

		if(T1 > tNear)
		{
			tNear = T1;
		}

		if(T2 < tFar)
		{
			tFar = T2;
		}
			

		if(tNear > tFar || tFar < 0)
		{
			return false;
		}
	}

	return true;
}