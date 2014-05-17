#ifndef CUDA_RAYTRACER_H_
#define CUDA_RAYTRACER_H_

#include <memory>
#include "curand_kernel.h"
#include "vector2.h"
#include "vector3.h"

class Camera;
class OpenGLSurface;
class Sphere;

class CudaRayTracer
{
private:
	bool use_pathtracing_;
	curandState *rand_state_d;
	Camera *camera_d;
	Vector3d *accumulation_buffer_d_;
	size_t current_sample_;	
	std::shared_ptr<OpenGLSurface> surface_;

	void init_curand(const std::shared_ptr<OpenGLSurface> &surface);
	void init_accumulation_buffer(const std::shared_ptr<OpenGLSurface> &surface);

	// private copy assignment operator and copy constructor
	CudaRayTracer& operator=(CudaRayTracer &other); 
	CudaRayTracer(CudaRayTracer &other);
public:
	CudaRayTracer();
	~CudaRayTracer();

	cudaError_t render(Camera &camera, Sphere **scene, int nSpheres);
	cudaError_t get_camera_distance(Camera &camera, const Vector2i &screen_coord, Sphere **scene, int n_spheres, double &dist_out);

	void set_surface(const std::shared_ptr<OpenGLSurface> &surface);
	void set_use_pathtracing(const bool value);

	size_t get_current_sample() const { return current_sample_; }
};

#endif