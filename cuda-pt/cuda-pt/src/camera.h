#ifndef CAMERA_H_
#define CAMERA_H_

#include "curand_kernel.h"
#include "error_assertion.h"
#include "vector3.h"
#include "vector2.h"
#include "ray.h"
#include <exception>
#include <random>

class Camera
{
private:
	Vector3d direction_;
	Vector3d position_;
	Vector3d up_;
	Vector3d n_;
	Vector3d u_;
	Vector3d v_;
	bool updated_this_frame_;

	double fov_;
	double aspect_ratio_;
	Vector2i resolution_;
	double focal_length_;

	double image_plane_height_;
	double image_plane_width_;
	double pixel_width_;
	double pixel_height_;
	Vector3d image_plane_start_;


	void calculate_n();
	void calculate_uv();

	
public:

	Camera(const Vector3d &pos, const Vector3d &dir, const Vector3d &up, const double fov, const double aspect_ratio, const Vector2i &resolution, double focal_length)
		: position_(pos), direction_(dir), up_(up), fov_(fov), aspect_ratio_(aspect_ratio), resolution_(resolution), focal_length_(focal_length) {

			if(aspect_ratio <= 0)
				throw std::exception("aspect_ratio must be greater than 0");
	}

	void set_position(const Vector3d &pos) {
		position_ = pos;
	}

	CUDA_CALLABLE const Vector3d position() const {
		return position_;
	}

	void set_direction(const Vector3d &dir) {
		direction_ = dir;
	}

	const Vector3d direction() const {
		return direction_;
	}

	CUDA_CALLABLE double focal_length() const {
		return focal_length_;
	}

	CUDA_CALLABLE void set_focal_length(double value) {
		focal_length_ = value;
	}

	bool updated_this_frame() const {
		return updated_this_frame_;
	}

	CUDA_CALLABLE void cast_ray(Ray &ray, int x, int y) const;
	CUDA_CALLABLE void cast_ray(Ray *ray, int x, int y) const;

	__device__ void cast_perturbed_ray(Ray &ray, int x, int y, double radius, curandState &rand_state) const;

	void update();

	void reset_update_flag();
};

#endif