#include "camera.h"
#include "vector2.h"
#include "mtrand.h"
#include "ray.h"

void Camera::CalculateN()
{
	n_ = direction_ * -1;
	n_.normalize();
}

void Camera::CalculateUV()
{
	u_ = up_.cross(n_);
	u_.normalize();

	v_ = n_.cross(u_);
	v_.normalize();

}

void Camera::update() {
	const int distance = 10;

	image_plane_height_ = 2 * distance * tan(fov_ / 2.0);
	image_plane_width_ = image_plane_height_ * aspect_ratio_;

	CalculateN();
	CalculateUV();

	Vector3d image_plane_center = position_ - n_ * distance;

	image_plane_start_ = image_plane_center + (u_ * (image_plane_width_ / 2.0)) - (v_ * (image_plane_height_ / 2.0));

	pixel_width_ = image_plane_width_ / resolution_.x();
	pixel_height_ = image_plane_height_ / resolution_.y();
}

void Camera::cast_ray(Ray *ray, int x, int y) const {
	ray->origin_ = position_;

	ray->direction_ = (image_plane_start_ - (u_ * pixel_width_ * (double)x) + (v_ * pixel_height_ * (double)y)) - position_;
	ray->direction_.normalize();
}

void Camera::cast_ray(Ray &ray, int x, int y) const {
	ray.origin_ = position_;

	ray.direction_ = (image_plane_start_ - (u_ * pixel_width_ * (double)x) + (v_ * pixel_height_ * (double)y)) - position_;
	ray.direction_.normalize();
}

__device__ void Camera::cast_perturbed_ray(Ray &ray, int x, int y, double radius, curandState &rand_state) const
{
	cast_ray(ray, x, y);

	Vector3d focus_point = position_ + ray.direction_ * focal_length_;

	double u_shift = curand_uniform(&rand_state); //(((double)mt_rand() - min) / range); //(MathUtil::get_rand()  * radius) - (radius / 2.0);
	double v_shift = curand_uniform(&rand_state); //(((double)mt_rand() - min) / range); //(MathUtil::get_rand()  * radius) - (radius / 2.0);

	double r = radius;

	ray.origin_ = position_ - (u_ * (r/2)) - (v_ * (r/2)) + (u_ * r * u_shift) + (v_ * r * v_shift);
	ray.direction_ = focus_point - ray.origin_;
	ray.direction_.normalize();
}