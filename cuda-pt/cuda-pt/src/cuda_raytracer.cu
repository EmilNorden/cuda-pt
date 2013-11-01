#include "cuda_raytracer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stack>

#include "Sphere.h"

#include "camera.h"

#include "intersectioninfo.h"
#include "opengl_surface.h"

#include <curand_kernel.h>
#include <chrono>

__device__ void trace(Ray &ray, Sphere *spheres, int nSpheres, int max_depth, Vector3d &color, curandState &rand_state);
__device__ void shade(Ray &ray, const IntersectionInfo &ii, Sphere *spheres, int index, int nSpheres, int max_depth, Vector3d &color, curandState &rand_state);

__device__ bool intersect(Ray &ray, const Sphere &sphere) {
	double num1 = sphere.position.x() - ray.origin_.x();
	double num2 = sphere.position.y() - ray.origin_.y();
	double num3 = sphere.position.z() - ray.origin_.z();
	double num4 = (num1 * num1 + num2 * num2 + num3 * num3);
	double num5 = sphere.radius * sphere.radius;
	
	double num6 = (num1 * ray.direction_.x() + num2 * ray.direction_.y() + num3 * ray.direction_.z());

	if (num6 < 0.0)
	return false;
	double num7 = num4 - num6 * num6;
	if (num7 > num5)
		return false;

	return true;
}

__device__ bool intersect(Ray &ray, const Sphere &sphere, IntersectionInfo  &ii) {
	double num1 = sphere.position.x() - ray.origin_.x();
	double num2 = sphere.position.y() - ray.origin_.y();
	double num3 = sphere.position.z() - ray.origin_.z();
	double num4 = (num1 * num1 + num2 * num2 + num3 * num3);
	double num5 = sphere.radius * sphere.radius;
	
	double num6 = (num1 * ray.direction_.x() + num2 * ray.direction_.y() + num3 * ray.direction_.z());

	if (num6 < 0.0)
	return false;
	double num7 = num4 - num6 * num6;
	if (num7 > num5)
		return false;

	double num8 = sqrt(num5 - num7);

	ii.distance = abs(num6 - num8);
	ii.coordinate = ray.origin_ + (ray.direction_ * ii.distance);
	ii.surface_normal = ii.coordinate - sphere.position;

	ii.surface_normal.normalize();


	return true;
}

__device__ int get_intersected_sphere(Ray &ray, Sphere *spheres, int nSpheres)
{
	int intersection_index = -1;
	double best_dist = DBL_MAX;
	for(int i = 0; i < nSpheres; ++i)
	{
		IntersectionInfo temp_ii;
		if(intersect(ray, spheres[i], temp_ii) && temp_ii.distance < best_dist)
		{
			best_dist = temp_ii.distance;
			intersection_index = i;
		}
	}

	return intersection_index;
}


__device__ int get_intersected_sphere(Ray &ray, Sphere *spheres, int nSpheres, IntersectionInfo &ii)
{
	int intersection_index = -1;
	ii.distance = DBL_MAX;
	for(int i = 0; i < nSpheres; ++i)
	{
		IntersectionInfo temp_ii;
		if(intersect(ray, spheres[i], temp_ii) && temp_ii.distance < ii.distance)
		{
			ii = temp_ii;
			intersection_index = i;
		}
	}

	return intersection_index;
}

__device__ void shade_light(Sphere &light, const IntersectionInfo &ii, Sphere &object, Vector3d &color)
{
	Vector3d dir = light.position - ii.coordinate;
	double inv_square_length = 1.0 / dir.length_squared();
	dir.normalize();
	double dot = dir.dot(ii.surface_normal);
	if(dot < 0)
		dot = 0;
	color += light.emissive * object.diffuse * dot * inv_square_length * (1 - object.refl_coeff);
}

__device__ void pathtrace(Ray &ray, const IntersectionInfo &ii, int sphere_index, Sphere *spheres, int nSpheres, int max_depth, Vector3d &color, curandState &rand_state)
{
	Vector3d random_dir = Vector3d::rand_unit_in_hemisphere(ii.surface_normal, rand_state);
	Ray random_ray;
	random_ray.refractive_index = ray.refractive_index;
	random_ray.depth_ = ray.depth_ + 1;
	random_ray.origin_ = ii.coordinate;
	random_ray.direction_ = random_dir;

	IntersectionInfo random_ii;

	int random_sphere_index = get_intersected_sphere(random_ray, spheres, nSpheres, random_ii);
	if(random_sphere_index != -1)
	{
		Vector3d viewer = ray.direction_ * -1;
		Vector3d halfway = (random_dir + viewer);
		halfway.normalize();

		double factor = halfway.dot(ii.surface_normal);

		Vector3d random_color;
		shade(random_ray, random_ii, spheres, random_sphere_index, nSpheres, max_depth, random_color, rand_state);
		color += spheres[sphere_index].diffuse * random_color * pow(factor, 2);
	}
}

__device__ void shade(Ray &ray, const IntersectionInfo &ii, Sphere *spheres, int index, int nSpheres, int max_depth, Vector3d &color, curandState &rand_state)
{
	//printf("Depth %d\n", ray.depth_);
	if(ray.depth_ > max_depth)
		return;

	color += spheres[index].emissive;

	// Path tracing
	pathtrace(ray, ii, index, spheres, nSpheres, max_depth, color, rand_state);

	//// Diffuse
	for(int i = 0; i < nSpheres; ++i)
	{
		if(i != index && !spheres[i].emissive.is_zero())
		{
			Ray shadow_ray;
			shadow_ray.origin_ = ii.coordinate;
			shadow_ray.direction_ = spheres[i].position - ii.coordinate;
			shadow_ray.direction_.normalize();

			if (get_intersected_sphere(shadow_ray, spheres, nSpheres) == i) //(intersect(shadow_ray, spheres[i]))
			{
				shade_light(spheres[i], ii, spheres[index], color);
			}
		}
	}

	// Reflection

	if(spheres[index].refl_coeff > 0)
	{
		Ray reflected_ray;
		reflected_ray.refractive_index = ray.refractive_index;
		reflected_ray.depth_ = ray.depth_ + 1;
		reflected_ray.origin_ = ii.coordinate;
		reflected_ray.direction_ = ray.direction_ - ii.surface_normal * 2.0 * ray.direction_.dot(ii.surface_normal);
		reflected_ray.direction_.normalize();

		Vector3d refl_color;
		trace(reflected_ray, spheres, nSpheres, max_depth, refl_color, rand_state);
		color += refl_color * spheres[index].refl_coeff;
	}
}

__device__ void trace(Ray &ray, Sphere *spheres, int nSpheres, int max_depth, Vector3d &color, curandState &rand_state)
{
	IntersectionInfo ii;
	int sphere_index = get_intersected_sphere(ray, spheres, nSpheres, ii);

	if(sphere_index != -1)
	{
		shade(ray, ii, spheres, sphere_index, nSpheres, max_depth, color, rand_state);
	}
}

__global__ void raytrace_kernel(int *ptr, Camera *camera, int depth, int sample, Vector3d *buffer_d, curandState *rand_states, const Vector2i *resolution_d, Sphere **scene, int nSpheres)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	curandState &rand_state = rand_states[index_y * resolution_d->x() + index_x];
	Vector3d color;
	Ray ray;
	ray.depth_ = 0;
	camera->cast_perturbed_ray(ray, index_x, index_y, 0.25, rand_state);

	Sphere *spheres = *scene;
	//Sphere spheres[4];
	//spheres[0].radius = 1;
	//spheres[0].position.x() = 0;
	//spheres[0].position.y() = 10;
	//spheres[0].position.z() = 0;
	//spheres[0].diffuse.x() = 1;
	//spheres[0].diffuse.y() = 1;
	//spheres[0].diffuse.z() = 1;
	//spheres[0].emissive = Vector3d(0.4, 0.4, 0.4) * 200;
	//spheres[0].refl_coeff = 0;

	//spheres[1].radius = 1.5;
	//spheres[1].position.x() = -4;
	//spheres[1].position.y() = 1.5;
	//spheres[1].position.z() = -5;
	//spheres[1].diffuse.x() = 1;
	//spheres[1].diffuse.y() = 1;
	//spheres[1].diffuse.z() = 1;
	//spheres[1].emissive = Vector3d(0.0, 0.0, 0.0);
	//spheres[1].refl_coeff = 1.0;

	//spheres[2].radius = 1.5;
	//spheres[2].position.x() = 0;
	//spheres[2].position.y() = 1.5;
	//spheres[2].position.z() = 5;
	//spheres[2].diffuse.x() = 1;
	//spheres[2].diffuse.y() = 0;
	//spheres[2].diffuse.z() = 0;
	//spheres[2].emissive = Vector3d(0.0, 0.0, 0.0);
	//spheres[2].refl_coeff = 0.25;

	//spheres[3].radius = 400;
	//spheres[3].position.x() = 0;
	//spheres[3].position.y() = -400;
	//spheres[3].position.z() = 0;
	//spheres[3].diffuse.x() = 1;
	//spheres[3].diffuse.y() = 1;
	//spheres[3].diffuse.z() = 1;
	//spheres[3].emissive = Vector3d(1.0, 1.0, 1.0) * 0.0;
	//spheres[3].refl_coeff = 0.0;

	//// Tiny red ball
	//spheres[4].radius = 0.25;
	//spheres[4].position.x() = -4;
	//spheres[4].position.y() = 1.5;
	//spheres[4].position.z() = -3.375;
	//spheres[4].diffuse.x() = 1;
	//spheres[4].diffuse.y() = 0;
	//spheres[4].diffuse.z() = 0;
	//spheres[4].emissive = Vector3d(0.0, 0.0, 0.0);
	//spheres[4].refl_coeff = 0.0;

	trace(ray, spheres, nSpheres, depth, color, rand_state);

	color.clamp(Vector3d(0, 0, 0), Vector3d(1, 1, 1));

	Vector3d &current_color = buffer_d[index_y * (resolution_d->x()) + index_x];
	current_color.multiply(sample);
	current_color += color;
	current_color.multiply(1 / (double)(sample + 1));
	ptr[index_y * resolution_d->x() + index_x] = 255 << 24 | static_cast<int>(current_color.x() * 255) << 16  | static_cast<int>(current_color.y() * 255) << 8 | static_cast<int>(current_color.z() * 255);
}

cudaError_t CudaRayTracer::render(Camera &camera, Sphere **scene, int nSpheres)
{
	dim3 block_size;
	block_size.x = 8;
	block_size.y = 8;

	dim3 grid_size;
	grid_size.x = surface_->resolution().x() / block_size.x;
	grid_size.y = surface_->resolution().y() / block_size.y;

	if(camera.updated_this_frame())
	{
		CUDA_CALL(cudaMemcpy(camera_d, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
		current_sample_ = 0;
	}

	surface_->map();
	raytrace_kernel<<<grid_size, block_size>>>(static_cast<int*>(surface_->pixel_buffer_object_d()), camera_d, 4, current_sample_++, accumulation_buffer_d_, rand_state_d, surface_->resolution_d(), scene, nSpheres);
	cudaError_t result = cudaDeviceSynchronize();
	surface_->unmap();

	return result;
}

__device__ double camera_dist_d;
__global__ void get_camera_distance_kernel(Camera *camera, int x, int y, Sphere **scene, int nSpheres)
{
	Ray ray;
	camera->cast_ray(ray, x, y); 
	IntersectionInfo ii;

	camera_dist_d = DBL_MAX;
	if(get_intersected_sphere(ray, *scene, nSpheres, ii) != -1)
	{
		camera_dist_d = ii.distance;
	}
}

cudaError_t CudaRayTracer::get_camera_distance(Camera &camera, const Vector2i &screen_coord, Sphere **scene, int n_spheres, double &dist_out)
{
	if(camera.updated_this_frame())
	{
		CUDA_CALL(cudaMemcpy(camera_d, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
		current_sample_ = 0;
	}


	get_camera_distance_kernel<<<1, 1>>>(camera_d, screen_coord.x(), screen_coord.y(), scene, n_spheres);

	double camera_dist;
	void *camera_dist_device_ptr;
	
	cudaMemcpyFromSymbol(&camera_dist, camera_dist_device_ptr, sizeof(camera_dist), 0, cudaMemcpyDeviceToHost);
	dist_out = camera_dist;
	return cudaDeviceSynchronize();
}

void CudaRayTracer::set_surface(const std::shared_ptr<OpenGLSurface> &surface)
{
	if(surface != nullptr && (surface_ == nullptr || surface->resolution() != surface_->resolution()))
	{
		init_curand(surface);
		init_accumulation_buffer(surface);
	}
	surface_ = surface;
}

__global__ void init_curand_kernel(curandState *state, unsigned long *seed, const Vector2i *resolution)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	curand_init(seed[index_y * resolution->x() + index_x], 0, 0, &state[index_y * resolution->x() + index_x]);
}

void CudaRayTracer::init_curand(const std::shared_ptr<OpenGLSurface> &surface)
{
	if(rand_state_d != nullptr)
	{
		CUDA_CALL(cudaFree(rand_state_d));
	}

	CUDA_CALL(cudaMalloc(&rand_state_d, sizeof(curandState) * surface->resolution().x() * surface->resolution().y()));
	
	unsigned long *seeds_d;

	// Randomly generate seeds for the kernels curand states.
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::minstd_rand0 generator (seed);

	unsigned long *seeds_h = new unsigned long[surface->resolution().x() * surface->resolution().y()];
	for(int i = 0; i < surface->resolution().x() * surface->resolution().y(); ++i)
	{
		seeds_h[i] = generator();
	}

	CUDA_CALL(cudaMalloc(&seeds_d, sizeof(unsigned long) * surface->resolution().x() * surface->resolution().y()));
	CUDA_CALL(cudaMemcpy(seeds_d, seeds_h, sizeof(unsigned long) * surface->resolution().x() * surface->resolution().y(),  cudaMemcpyHostToDevice));
	delete[] seeds_h;
	
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	dim3 grid_size;
	grid_size.x = surface->resolution().x() / block_size.x;
	grid_size.y = surface->resolution().y() / block_size.y;

	init_curand_kernel<<<grid_size, block_size>>>(rand_state_d, seeds_d, surface->resolution_d());

	cudaDeviceSynchronize();
}

CudaRayTracer::CudaRayTracer()
	: rand_state_d(nullptr), accumulation_buffer_d_(nullptr), current_sample_(0)
{
	CUDA_CALL(cudaMalloc(&camera_d, sizeof(Camera)));
}

CudaRayTracer::~CudaRayTracer()
{
	CUDA_CALL(cudaFree(camera_d));

	if(accumulation_buffer_d_ != nullptr)
	{
		CUDA_CALL(cudaFree(accumulation_buffer_d_));
	}

	if(rand_state_d != nullptr)
	{
		CUDA_CALL(cudaFree(rand_state_d));
	}
}

void CudaRayTracer::init_accumulation_buffer(const std::shared_ptr<OpenGLSurface> &surface)
{
	if(accumulation_buffer_d_ != nullptr)
	{
		CUDA_CALL(cudaFree(accumulation_buffer_d_));
	}

	CUDA_CALL(cudaMalloc(&accumulation_buffer_d_, sizeof(Vector3d) * surface->resolution().x() * surface->resolution().y()));
}