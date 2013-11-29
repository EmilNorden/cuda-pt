#include "camera.h"
#include "vector2.h"
#include "ray.h"

#define PI 3.14159265359

void Camera::calculate_n()
{
	n_ = direction_ * -1;
	n_.normalize();
}

void Camera::calculate_uv()
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

	calculate_n();
	calculate_uv();

	Vector3d image_plane_center = position_ - n_ * distance;

	image_plane_start_ = image_plane_center + (u_ * (image_plane_width_ / 2.0)) - (v_ * (image_plane_height_ / 2.0));

	pixel_width_ = image_plane_width_ / resolution_.x();
	pixel_height_ = image_plane_height_ / resolution_.y();
	updated_this_frame_ = true;
}

void Camera::cast_ray(Ray *ray, int x, int y) const {
	ray->origin_ = position_;

	ray->direction_ = (image_plane_start_ - (u_ * pixel_width_ * (double)x) + (v_ * pixel_height_ * (double)y)) - position_;
	ray->direction_.normalize_device();
}

void Camera::cast_ray(Ray &ray, int x, int y) const {
	ray.origin_ = position_;

	ray.direction_ = (image_plane_start_ - (u_ * pixel_width_ * (double)x) + (v_ * pixel_height_ * (double)y)) - position_;
	ray.direction_.normalize_device();
}

__device__ void Camera::cast_perturbed_ray(Ray &ray, int x, int y, double radius, curandState &rand_state) const
{
	cast_ray(ray, x, y);

	Vector3d focus_point = position_ + ray.direction_ * focal_length_;

	double angle = curand_uniform(&rand_state) * PI * 2;
	double length = curand_uniform(&rand_state) * radius;

	ray.origin_ = position_ + (u_ * sin(angle) * length) + (v_ * cos(angle) * length);
	ray.direction_ = focus_point - ray.origin_;
	ray.direction_.normalize_device();

	// This created a rectangular blur
	/*double u_shift = curand_uniform(&rand_state);
	double v_shift = curand_uniform(&rand_state); 

	double r = radius;

	ray.origin_ = position_ - (u_ * (r/2)) - (v_ * (r/2)) + (u_ * r * u_shift) + (v_ * r * v_shift);
	ray.direction_ = focus_point - ray.origin_;
	ray.direction_.normalize();*/
}

void Camera::reset_update_flag()
{
	updated_this_frame_ = false;
}