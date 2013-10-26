
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stack>

#include "src/mtrand.h"
#include "src/camera.h"

#include "src/test.h"
#include "src/intersectioninfo.h"

#include "src/static_stack.h"
#include "src/static_uniform_heap.h"


class Sphere
{
public:
	double radius;
	Vector3d position;
	Vector3d emissive;
	Vector3d diffuse;
	double refl_coeff;
};

__device__ void ray_trace(Ray &ray, Sphere *spheres, int nSpheres, int max_depth, Vector3d &color);

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

__device__ bool intersect(Ray *ray, const Sphere &sphere) {
	double num1 = sphere.position.x() - ray->origin_.x();
	double num2 = sphere.position.y() - ray->origin_.y();
	double num3 = sphere.position.z() - ray->origin_.z();
	double num4 = (num1 * num1 + num2 * num2 + num3 * num3);
	double num5 = sphere.radius * sphere.radius;
	
	double num6 = (num1 * ray->direction_.x() + num2 * ray->direction_.y() + num3 * ray->direction_.z());

	if (num6 < 0.0)
	return false;
	double num7 = num4 - num6 * num6;
	if (num7 > num5)
		return false;

	double num8 = sqrt(num5 - num7);

	ray->intersection.distance = abs(num6 - num8);
	ray->intersection.coordinate = ray->origin_ + (ray->direction_ * ray->intersection.distance);
	ray->intersection.surface_normal = ray->intersection.coordinate - sphere.position;

	ray->intersection.surface_normal.normalize();


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

__device__ void shade(Ray &ray, const IntersectionInfo &ii, Sphere *spheres, int index, int nSpheres, int max_depth, Vector3d &color)
{
	//printf("Depth %d\n", ray.depth_);
	if(ray.depth_ > max_depth)
		return;

	color += spheres[index].emissive;

	//// To be replaced with path tracing
	color += Vector3d(0.1, 0.1, 0.1);

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

		ray_trace(reflected_ray, spheres, nSpheres, max_depth, color);
	}
}

__device__ void ray_trace(Ray &ray, Sphere *spheres, int nSpheres, int max_depth, Vector3d &color)
{
	IntersectionInfo ii;
	int sphere_index = get_intersected_sphere(ray, spheres, nSpheres, ii);

	if(sphere_index != -1)
	{
		//color = best_ii.surface_normal;
		shade(ray, ii, spheres, sphere_index, nSpheres, max_depth, color);
	}
}

__global__ void dostuff2(int *ptr, Camera *camera, int depth, Vector3d *light_d, int sample, Vector3d *buffer_d)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	Vector3d color;
	Ray ray;
	ray.depth_ = 0;
	camera->cast_ray(ray, index_x, index_y);

	Sphere spheres[2];
	spheres[0].radius = 10;
	spheres[0].position.x() = light_d->x();
	spheres[0].position.y() = light_d->y();
	spheres[0].position.z() = light_d->z();
	spheres[0].diffuse.x() = 1;
	spheres[0].diffuse.y() = 1;
	spheres[0].diffuse.z() = 1;
	spheres[0].emissive = Vector3d(0.4, 0.4, 0.4) * 400;
	spheres[0].refl_coeff = 0;

	spheres[1].radius = 10;
	spheres[1].position.x() = 11;
	spheres[1].position.y() = 0;
	spheres[1].position.z() = -80;
	spheres[1].diffuse.x() = 1;
	spheres[1].diffuse.y() = 1;
	spheres[1].diffuse.z() = 1;
	spheres[1].emissive = Vector3d(0.0, 0.0, 0.0);
	spheres[1].refl_coeff = 0.0;

	spheres[2].radius = 5;
	spheres[2].position.x() = 11;
	spheres[2].position.y() = 0;
	spheres[2].position.z() = -70;
	spheres[2].diffuse.x() = 1;
	spheres[2].diffuse.y() = 0;
	spheres[2].diffuse.z() = 0;
	spheres[2].emissive = Vector3d(0.0, 0.0, 0.0);
	spheres[2].refl_coeff = 0.0;

	ray_trace(ray, spheres, 3, depth, color);
	//ray_trace(camera, depth, color);

	color.clamp(Vector3d(0, 0, 0), Vector3d(1, 1, 1));

	Vector3d &current_color = buffer_d[index_y * (800) + index_x];
	current_color.multiply(sample);
	current_color += color;
	current_color.multiply(1 / (double)(sample + 1));
	ptr[index_y * 800 + index_x] = 255 << 24 | static_cast<int>(current_color.x() * 255) << 16  | static_cast<int>(current_color.y() * 255) << 8 | static_cast<int>(current_color.z() * 255);


	//float r = ptr[index_y * (800) + index_x];
	//float g = ptr[index_y * (800) + index_x + 1];
	//float b = ptr[index_y * (800) + index_x + 2];
	//
	//r *= sample;
	//g *= sample;
	//b *= sample;

	//r += color.x();
	//g += color.y();
	//b += color.z();

	//r /= sample + 1;
	//g /= sample + 1;
	//b /= sample + 1;

	/*int buffer_color = ptr[index_y * 800 + index_x];
	int r = (buffer_color & 0x00FF0000) >> 16;
	int g = (buffer_color & 0x0000FF00) >> 8;
	int b = (buffer_color & 0x000000FF0);
	
	r *= sample;
	g *= sample;
	b *= sample;

	r += static_cast<int>(color.x() * 255);
	g += static_cast<int>(color.y() * 255);
	b += static_cast<int>(color.z() * 255);

	r /= sample + 1;
	g /= sample + 1;
	b /= sample + 1;

	if(r > 255)
		r = 255;
	else if(r < 0)
		r = 0;
	if(g > 255)
		g = 255;
	else if(g < 0)
		g = 0;
	if(b > 255)
		b = 255;
	else if(b < 0)
		b = 0;

	
	ptr[index_y * 800 + index_x] = 255 << 24 | r << 16  | g << 8 | b;*/
	//ptr[index_y * 800 + index_x] = 255 << 24 | static_cast<int>(color.x() * 255) << 16  | static_cast<int>(color.y() * 255) << 8 | static_cast<int>(color.z() * 255);
}

__global__ void dostuffKernel(int *ptr, Camera *camera, int seed, int depth)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	StaticUniformHeap<Ray, 20> heap;

	Vector3d color;
	
	Sphere s;
	s.radius = 10;
	s.position.x() = 0;
	s.position.y() = 0;
	s.position.z() = -50;
	s.diffuse.x() = s.diffuse.y() = s.diffuse.z() = 1;
	s.emissive = Vector3d(0.1, 0.1, 0.1);
	s.refl_coeff = 1;

	
	Ray *ray = heap.allocate();
	camera->cast_ray(ray, index_x, index_y);
	
	{
		StaticStack<Ray*, 20> rays;

		rays.push(ray);
		while(!rays.empty())
		{
			Ray* current_ray = rays.pop();

			if(current_ray->depth_ < depth)
			{
				if(intersect(current_ray, s))
				{
					color += s.emissive;

					if(!s.diffuse.is_zero())
					{
	//					// Loopa över alla andra emissive objects och utför ljusberäkning
					}

					if(s.refl_coeff)
					{
						Ray *refl_ray = heap.allocate();
						current_ray->reflected = refl_ray;
						
						//refl_ray->depth_ = 50;
						//int dephtesku = current_ray->depth_ + 1;
						//refl_ray->depth_ = current_ray->depth_ + 1;

						//refl_ray->origin_ = current_ray->intersection.coordinate;
	//					refl_ray->direction_ = current_ray->direction_ - current_ray->intersection.surface_normal * 2.0 * current_ray->direction_.dot(current_ray->intersection.surface_normal);
	//					refl_ray->direction_.normalize();
	//					current_ray->reflected = refl_ray;
	//					rays.push(refl_ray);
					}
				}
			}
		}
	}

	//delete ray;

	/*Stack<Ray*> to_examine(10);
	Stack<Ray*> to_traverse(10);

	to_examine.push(&ray);*/

	/*while(!to_examine.empty())
	{
		Ray *ray = to_examine.pop();
		
		if(ray->reflected != nullptr)
		{
			to_examine.push(ray->reflected);
			to_traverse.push(ray->reflected);
		}
	}*/
	
	//else if(intersect(ray, s2))
	//{
		//b = 255;
	//}

	ptr[index_y * 800 + index_x] = 255 << 24 | static_cast<int>(color.x() * 255) << 16  | static_cast<int>(color.y() * 255) << 8 | static_cast<int>(color.z() * 255);

	

}

cudaError_t dostuff(void *ptr, Camera *device_camera, int seed, Vector3d *light_d, int sample, Vector3d *buffer_d)
{
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	dim3 grid_size;
	grid_size.x = 800 / block_size.x;
	grid_size.y = 600 / block_size.y;

		

	//dostuffKernel<<<grid_size, block_size>>>(static_cast<int*>(ptr), device_camera, 0, 1);
	dostuff2<<<grid_size, block_size>>>(static_cast<int*>(ptr), device_camera, 1, light_d, sample, buffer_d);
	//dostuffKernel<<<grid_size, block_size>>>(static_cast<int*>(ptr), device_camera, seed, 4);

	return cudaDeviceSynchronize();
}

__global__ void clearBufferKernel(Vector3d *ptr)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	Vector3d &color = ptr[index_y * (800) + index_x];
	color.x() = color.y() = color.z() = 0;
}

cudaError_t clearBuffer(Vector3d *ptr)
{
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	dim3 grid_size;
	grid_size.x = 800 / block_size.x;
	grid_size.y = 600 / block_size.y;

	clearBufferKernel<<<grid_size, block_size>>>(ptr);

	return cudaDeviceSynchronize();
}