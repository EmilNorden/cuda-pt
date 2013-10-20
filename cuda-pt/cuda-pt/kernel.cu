
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "src/camera.h"

#include "src/test.h"

class Sphere
{
public:
	double radius;
	double x, y, z;
	double r, g, b;
};

__device__ bool intersect(const Ray &ray, const Sphere &sphere) {
	double num1 = sphere.x - ray.origin_.x();
	double num2 = sphere.y - ray.origin_.y();
	double num3 = sphere.z - ray.origin_.z();
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

__global__ void dostuffKernel(int *ptr, Camera *camera)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = 0;
	int g = 0;
	int b = 0;
	int a = 255;

	const Vector3d campos = camera->position();
	
	Ray ray;
	camera->cast_ray(ray, index_x, index_y);



	Sphere s;
	s.radius = 10;
	s.x = 0;
	s.y = 0;
	s.z = -50;
	s.r = s.g = s.b = 255;

	if(intersect(ray, s))
		r = g = b = 255;


	

	ptr[index_y * 800 + index_x] = a << 24 | r << 16  | g << 8 | b;

	

}

cudaError_t dostuff(void *ptr, Camera *device_camera)
{
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	dim3 grid_size;
	grid_size.x = 800 / block_size.x;
	grid_size.y = 600 / block_size.y;

	dostuffKernel<<<grid_size, block_size>>>(static_cast<int*>(ptr), device_camera);

	return cudaDeviceSynchronize();
}