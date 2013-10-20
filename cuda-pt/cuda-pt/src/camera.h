#ifndef CAMERA_H_
#define CAMERA_H_

#include "cuda_helpers.h"
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

	double fov_;
	double aspect_ratio_;
	Vector2i resolution_;
	double focal_length_;

	double image_plane_height_;
	double image_plane_width_;
	double pixel_width_;
	double pixel_height_;
	Vector3d image_plane_start_;


	void CalculateN();
	void CalculateUV();
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

	const Vector3d direction() const {
		return direction_;
	}

	const double focal_length() const {
		return focal_length_;
	}

	void set_focal_length(double value) {
		focal_length_ = value;
	}

	CUDA_CALLABLE void cast_ray(Ray &ray, int x, int y) const;

	void cast_perturbed_ray(Ray &ray, int x, int y, double radius, std::shared_ptr<std::mt19937> &mt_rand) const;

	void update();
};

#endif