#include "cuda_raytracer.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stack>

#include "Sphere.h"

#include "camera.h"

#include "intersectioninfo.h"
#include "opengl_surface.h"

#include <curand_kernel.h>
#include <chrono>

cudaStream_t stream1, stream2, stream3, stream4;

__device__ void trace(const Ray &ray, const Sphere *spheres, const int nSpheres, const int max_depth, Vector3d &color, curandState &rand_state, const double branch_factor, bool use_pt);
__device__ void shade(const Ray &ray, const IntersectionInfo &ii, const Sphere *spheres, const unsigned int index, const int nSpheres, const int max_depth, Vector3d &color, curandState &rand_state, const double branch_factor, bool use_pt);
//
//__device__ bool intersect(Ray &ray, const Sphere &sphere, IntersectionInfo &ii) {
//	double a = ray.direction_.dot(ray.direction_);
//	double b = ray.direction_.dot((ray.origin_ - sphere.position) * 2.0);
//	double c = sphere.position.dot(sphere.position) + ray.origin_.dot(ray.origin_) - (ray.origin_.dot(sphere.position) * 2.0) - sphere.radius * sphere.radius;
//	double D = b * b + (-4.0) * a * c;
//
//	if(D < 0)
//		return false;
//
//	D = sqrt(D);
//
//	double t = (-0.5)*(b+D)/a;
//	if(t > 0.0)
//	{
//		ii.distance = sqrt(a)*t;
//		ii.coordinate = ray.origin_ + ray.direction_*t;
//		ii.surface_normal = (ii.coordinate - sphere.position) / sphere.radius;
//	}
//	else
//	{
//		return false;
//	}
//
//	return true;
//}

__device__ bool intersect(const Ray &ray, const Sphere &sphere, IntersectionInfo  &ii) {
	double num1 = __dsub_rn(sphere.position.x(), ray.origin_.x());
	double num2 = __dsub_rn(sphere.position.y(), ray.origin_.y());
	double num3 = __dsub_rn(sphere.position.z(), ray.origin_.z());
	double num4 = __fma_rn(num1, num1, __fma_rn(num2, num2, __dmul_rn(num3, num3)));
	double num5 = __dmul_rn(sphere.radius, sphere.radius);
	
	double num6 = __fma_rn(num1, ray.direction_.x(), 
					__fma_rn(num2, ray.direction_.y(), __dmul_rn(num3, ray.direction_.z())));

	if (num6 < 0.0)
	return false;
	double num7 = num4 - num6 * num6;
	if (num7 > num5)
		return false;

	ii.distance = abs(num6 - sqrt(num5 - num7));
	ii.object_pos = sphere.position;
	//ii.coordinate = ray.origin_ + (ray.direction_ * ii.distance);
	//ii.surface_normal = ii.coordinate - sphere.position;

	//ii.surface_normal.normalize_device();


	return true;
}

__device__ int get_intersected_sphere(const Ray &ray, const Sphere *spheres, const int nSpheres)
{
	int intersection_index = -1;
	double best_dist = DBL_MAX;
	for(int i = 0; i < nSpheres; ++i)
	{
		IntersectionInfo temp_ii;
		if(intersect(ray, spheres[i], temp_ii) && temp_ii.distance < best_dist) //if(spheres[i].intersect(ray, temp_ii) && temp_ii.distance < best_dist)
		{
			best_dist = temp_ii.distance;
			intersection_index = i;
		}
	}

	return intersection_index;
}


__device__ int get_intersected_sphere(const Ray &ray, const Sphere *spheres, const int nSpheres, IntersectionInfo &ii)
{
	int intersection_index = 0;
	ii.distance = DBL_MAX;
	for(int i = 0; i < nSpheres; ++i)
	{
		IntersectionInfo temp_ii;
		if(intersect(ray, spheres[i], temp_ii) && temp_ii.distance < ii.distance) //if(spheres[i].intersect(ray, temp_ii) && temp_ii.distance < ii.distance)
		{
			ii = temp_ii;
			intersection_index = i + 1;
		}
	}

	ii.coordinate = ray.origin_ + (ray.direction_ * ii.distance);
	ii.surface_normal = ii.coordinate - ii.object_pos;

	ii.surface_normal.normalize_device();

	return intersection_index;
}

__device__ void shade_light(const Sphere &light, const IntersectionInfo &ii, const Sphere &object, Vector3d &color)
{
	Vector3d dir = light.position - ii.coordinate;
	double inv_square_length = 1.0 / dir.length_squared();
	dir.normalize_device();
	double dot = dir.dot_device(ii.surface_normal);
	if(dot < 0)
		dot = 0;
	color += light.emissive * object.diffuse * dot * inv_square_length ;//* (1-object.refl_coeff);
}

__device__ void pathtrace(const Ray &ray, const IntersectionInfo &ii, const unsigned int sphere_index, const Sphere *spheres, const int nSpheres, const int max_depth, Vector3d &color, curandState &rand_state, const double branch_factor)
{
	Vector3d random_dir = Vector3d::rand_unit_in_hemisphere(ii.surface_normal, rand_state);
	Ray random_ray;
	random_ray.refractive_index = ray.refractive_index;
	random_ray.depth_ = ray.depth_ + 1;
	random_ray.origin_ = ii.coordinate;
	random_ray.direction_ = random_dir;

	float importance = 1; // random_dir.dot(ii.surface_normal);

	IntersectionInfo random_ii;

	int random_sphere_index = get_intersected_sphere(random_ray, spheres, nSpheres, random_ii);

	Vector3d viewer = ray.direction_ * -1;
	Vector3d halfway = (random_dir + viewer);
	halfway.normalize_device();

	Vector3d random_color;
	shade(random_ray, random_ii, spheres, random_sphere_index - 1, nSpheres, max_depth, random_color, rand_state, branch_factor, true);

	Vector3d dir = random_ii.coordinate - ii.coordinate;
	color += spheres[sphere_index].diffuse * random_color;// * (1-spheres[sphere_index].refl_coeff);
}

__device__ void shade(const Ray &ray, const IntersectionInfo &ii, const Sphere *spheres, const unsigned int index, const int nSpheres, const int max_depth, Vector3d &color, curandState &rand_state, const double branch_factor, bool use_pt)
{
	const unsigned int sphere_index = index % nSpheres;
	if(ray.depth_ > max_depth)
		return;

	color += spheres[sphere_index].emissive;

	

	if(branch_factor > spheres[sphere_index].refl_coeff)
	{
		// Path tracing
		if(use_pt)
			pathtrace(ray, ii, sphere_index, spheres, nSpheres, max_depth, color, rand_state, branch_factor);
		//// Diffuse
		for(int i = 0; i < nSpheres; ++i)
		{
			if(i != sphere_index	&& !spheres[i].emissive.is_zero())
			{
				Ray shadow_ray;
				shadow_ray.origin_ = ii.coordinate;
				shadow_ray.direction_ = spheres[i].position - ii.coordinate;
				shadow_ray.direction_.normalize_device();

				if (get_intersected_sphere(shadow_ray, spheres, nSpheres) == i)
				{
					shade_light(spheres[i], ii, spheres[sphere_index], color);
				}
			}
		}
	}
	else
	{
		// Reflection
		Ray reflected_ray;
		reflected_ray.refractive_index = ray.refractive_index;
		reflected_ray.depth_ = ray.depth_ + 1;
		reflected_ray.origin_ = ii.coordinate;
		reflected_ray.direction_ = ray.direction_ - ii.surface_normal * 2.0 * ray.direction_.dot_device(ii.surface_normal);
		reflected_ray.direction_.normalize_device();

		Vector3d refl_color;
		trace(reflected_ray, spheres, nSpheres, max_depth, refl_color, rand_state, branch_factor, use_pt);
		color += refl_color;// * spheres[index].refl_coeff;
	}
}

__device__ void trace(const Ray &ray, const Sphere *spheres, const int nSpheres, const int max_depth, Vector3d &color, curandState &rand_state, const double branch_factor, bool use_pt)
{
	IntersectionInfo ii;
	int sphere_index = get_intersected_sphere(ray, spheres, nSpheres, ii);

	shade(ray, ii, spheres, sphere_index - 1, nSpheres, max_depth, color, rand_state, branch_factor, use_pt);
	color.multiply(((sphere_index | (~sphere_index + 1)) >> 31) & 1);
	
}

__global__ void raytrace_kernel(int *ptr, Camera *camera, int depth, int sample, Vector3d *buffer_d, curandState *rand_states, const Vector2i *resolution_d, Sphere **scene, int nSpheres, int y_offset, bool use_pt)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	index_y += y_offset;

	curandState &rand_state = rand_states[index_y * resolution_d->x() + index_x];
	Sphere *spheres = *scene;

	__shared__ Sphere spheres2[64];
	memcpy(spheres2, *scene, sizeof(Sphere) * nSpheres);
	
	Vector3d color;
	Ray ray;
	ray.depth_ = 0;
	camera->cast_perturbed_ray(ray, index_x, index_y, rand_state);

	

	double branch_factor = curand_uniform(&rand_state);
	//clock_t start = clock();
	trace(ray, spheres2, nSpheres, depth, color, rand_state, branch_factor, use_pt);
	//clock_t end = clock();
	
	//double factor = (end - start) / 1000.0;
	//Vector3d current_color(factor, 0, 1 - factor);

	color.clamp(Vector3d(0, 0, 0), Vector3d(1, 1, 1));


	Vector3d &current_color = buffer_d[index_y * (resolution_d->x()) + index_x];
	current_color.multiply(sample);
	current_color += color;
	current_color.multiply(__drcp_rn((double)(sample + 1)));
	ptr[index_y * resolution_d->x() + index_x] = 255 << 24 | static_cast<int>(current_color.x() * 255) << 16  | static_cast<int>(current_color.y() * 255) << 8 | static_cast<int>(current_color.z() * 255);
}

__global__ void raytrace_interpolation_kernel(int *ptr, Camera *camera, int depth, int sample, Vector3d *buffer_d, curandState *rand_states, const Vector2i *resolution_d, Sphere **scene, int nSpheres, int y_offset, bool use_pt)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	index_y += y_offset;

	int index = index_y * resolution_d->x() + index_x;
	//
	curandState &rand_state = rand_states[index];
	Vector3d color;
	Ray ray;
	ray.depth_ = 0;
	camera->cast_perturbed_ray(ray, index_x, index_y, rand_state);

	Sphere *spheres = *scene;

	double branch_factor = curand_uniform(&rand_state);
	//clock_t start = clock();
	trace(ray, spheres, nSpheres, depth, color, rand_state, branch_factor, use_pt);
	////clock_t end = clock();
	//
	//double factor = (end - start) / 1000.0;
	//Vector3d current_color(factor, 0, 1 - factor);

	color.clamp(Vector3d(0, 0, 0), Vector3d(1, 1, 1));


	Vector3d &current_color = buffer_d[index];
	current_color.multiply(sample);
	current_color += color;
	current_color.multiply(__drcp_rn((double)(sample + 1)));
	ptr[index] = 255 << 24 | static_cast<int>(current_color.x() * 255) << 16  | static_cast<int>(current_color.y() * 255) << 8 | static_cast<int>(current_color.z() * 255);
}

cudaError_t CudaRayTracer::render(Camera &camera, Sphere **scene, int nSpheres)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y =  16;

	dim3 grid_size;
	grid_size.x = surface_->resolution().x() / block_size.x;
	grid_size.y = (surface_->resolution().y() / 1) / block_size.y;

	if(camera.updated_this_frame())
	{
		CUDA_CALL(cudaMemcpy(camera_d, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
		current_sample_ = 0;
		camera.reset_update_flag();
	}

	surface_->map();
	raytrace_interpolation_kernel<<<grid_size, block_size, 0, stream1>>>(static_cast<int*>(surface_->pixel_buffer_object_d()), camera_d, 2, current_sample_++, accumulation_buffer_d_, rand_state_d, surface_->resolution_d(), scene, nSpheres, 0, use_pathtracing_);
	//raytrace_kernel<<<grid_size, block_size, 0, stream2>>>(static_cast<int*>(surface_->pixel_buffer_object_d()), camera_d, 4, current_sample_++, accumulation_buffer_d_, rand_state_d, surface_->resolution_d(), scene, nSpheres, 240);
	//raytrace_kernel<<<grid_size, block_size, 0, stream3>>>(static_cast<int*>(surface_->pixel_buffer_object_d()), camera_d, 4, current_sample_, accumulation_buffer_d_, rand_state_d, surface_->resolution_d(), scene, nSpheres, 240);
	//raytrace_kernel<<<grid_size, block_size, 0, stream4>>>(static_cast<int*>(surface_->pixel_buffer_object_d()), camera_d, 4, current_sample_++, accumulation_buffer_d_, rand_state_d, surface_->resolution_d(), scene, nSpheres, 360);
	cudaError_t result = cudaDeviceSynchronize();
	surface_->unmap();

	return result;
}
double *distance_d = nullptr;
__device__ double camera_dist_d;
__global__ void get_camera_distance_kernel(Camera *camera, int x, int y, Sphere **scene, int nSpheres, double *distance_d)
{
	Ray ray;
	camera->cast_ray(ray, x, y); 
	IntersectionInfo ii;

	camera_dist_d = DBL_MAX;
	if(get_intersected_sphere(ray, *scene, nSpheres, ii) != -1)
	{
		*distance_d = ii.distance;
	}
}


cudaError_t CudaRayTracer::get_camera_distance(Camera &camera, const Vector2i &screen_coord, Sphere **scene, int n_spheres, double &dist_out)
{
	if(camera.updated_this_frame())
	{
		CUDA_CALL(cudaMemcpy(camera_d, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
		current_sample_ = 0;
	}

	if(!distance_d)
	{
		cudaMalloc(&distance_d, sizeof(double));
	}

	get_camera_distance_kernel<<<1, 1>>>(camera_d, screen_coord.x(), screen_coord.y(), scene, n_spheres, distance_d);

	cudaMemcpy(&dist_out, distance_d, sizeof(double), cudaMemcpyDeviceToHost);

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
	: rand_state_d(nullptr), accumulation_buffer_d_(nullptr), current_sample_(0), use_pathtracing_(true)
{
	CUDA_CALL(cudaMalloc(&camera_d, sizeof(Camera)));

	CUDA_CALL(cudaStreamCreate(&stream1));
	CUDA_CALL(cudaStreamCreate(&stream2));
	CUDA_CALL(cudaStreamCreate(&stream3));
	CUDA_CALL(cudaStreamCreate(&stream4));
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
	CUDA_CALL(cudaMemset(accumulation_buffer_d_, 0, sizeof(Vector3d) * surface->resolution().x() * surface->resolution().y()));
}

void CudaRayTracer::set_use_pathtracing(const bool value)
{
	use_pathtracing_ = value;
}