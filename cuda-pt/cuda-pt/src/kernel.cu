
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stack>

#include "Sphere.h"
#include "octree.h"

#include "camera.h"

#include "intersectioninfo.h"

#include <curand_kernel.h>

#include <ctime>

//__global__ void setup_scene_kernel(Sphere **scene, int *nSpheres)
//{
//	Sphere *spheres = new Sphere[66];
//
//	double angle_step = (3.14159265359 * 2) / 4.0;
//	for(int i = 0; i < 8; ++i)
//	{
//		for(int j = 0; j < 8; ++j)
//		{
//			spheres[i + (j * 8) + 2].radius = 1;
//			spheres[i + (j * 8) + 2].position.x() = -6 + i*2;
//			spheres[i + (j * 8) + 2].position.y() = 3 + (sin((i + (j * 8) / 63.0) * 4)) * 0.3;
//			spheres[i + (j * 8) + 2].position.z() = -6 + j*2;
//			spheres[i + (j * 8) + 2].diffuse.x() = (i / 7.0);
//			spheres[i + (j * 8) + 2].diffuse.y() = 1;
//			spheres[i + (j * 8) + 2].diffuse.z() = (j / 7.0);
//			spheres[i + (j * 8) + 2].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//			spheres[i + (j * 8) + 2].refl_coeff = 0;
//
//			if(((i + (j * 8) + 2) % 2) == 0)
//			{
//				spheres[i + (j * 8) + 2].refl_coeff = 0.5;
//			}
//			//if((i == 0 || i == 7) && (j == 0 || j == 7))
//			//{
//			//	spheres[i + (j * 8) + 2].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//			//}
//			//else
//			//{
//			//	spheres[i + (j * 8) + 2].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//			//}
//		}
//	}
//	spheres[0].radius = 400;
//	spheres[0].position.x() = 0;
//	spheres[0].position.y() = -400;
//	spheres[0].position.z() = 0;
//	spheres[0].diffuse.x() = 1;//0.5;
//	spheres[0].diffuse.y() = 1;
//	spheres[0].diffuse.z() = 1;//0.5;
//	spheres[0].emissive = Vector3d(1.0, 0.5, 0.5) * 0;
//	spheres[0].refl_coeff = 0.00;
//
//	spheres[1].radius = 20;
//	spheres[1].position.x() = 0;
//	spheres[1].position.y() = 30;
//	spheres[1].position.z() = -60;
//	spheres[1].diffuse.x() = 0.5;
//	spheres[1].diffuse.y() = 1;
//	spheres[1].diffuse.z() = 0.5;
//	spheres[1].emissive = Vector3d(1.0, 0.8, 0.8) * 600;
//	spheres[1].refl_coeff = 0.00;
//
//	*scene = spheres;
//	*nSpheres = 66;
//}

__global__ void setup_octree_kernel(Sphere **spheres, int nspheres, Octree **octree)
{
	printf("is this even?\n");
	Octree *tree = new Octree;
	for(int i = 0; i < nspheres; ++i)
		tree->spheres.add((*spheres)[i]);

	tree->build();
	*octree = tree;
}

//__global__ void setup_scene_kernel(Sphere **scene, int *nSpheres)
//{
//	Sphere *spheres = new Sphere[9];
//	
//	double angle_step = (3.14159265359 * 2) / 4.0;
//	for(int i = 0; i < 4; ++i)
//	{
//		double angle = angle_step * (i-2);
//		double x = sin(angle) * 3;
//		double z = cos(angle) * 3;
//
//		spheres[i].radius = 1.5;
//		spheres[i].position.x() = x;
//		spheres[i].position.y() = 1.5;
//		spheres[i].position.z() = z;
//		spheres[i].diffuse.x() = 1;
//		spheres[i].diffuse.y() = 1;
//		spheres[i].diffuse.z() = 1;
//		spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//		spheres[i].refl_coeff = 0.0;
//	}
//
//	for(int i = 5; i < 7; ++i)
//	{
//		double angle = angle_step * (i-0) + (angle_step / 2.0);
//		double x = sin(angle) * 5;
//		double z = cos(angle) * 5;
//
//		spheres[i].radius = 0.5;
//		spheres[i].position.x() = x;
//		spheres[i].position.y() = 2.5;
//		spheres[i].position.z() = z;
//		spheres[i].diffuse.x() = 1;
//		spheres[i].diffuse.y() = 1;
//		spheres[i].diffuse.z() = 1;
//		spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 17.5;
//		spheres[i].refl_coeff = 0.75;
//	}
//
//	spheres[4].radius = 400;
//	spheres[4].position.x() = 0;
//	spheres[4].position.y() = -400;
//	spheres[4].position.z() = 0;
//	spheres[4].diffuse.x() = 1;
//	spheres[4].diffuse.y() = 1;
//	spheres[4].diffuse.z() = 1;
//	spheres[4].emissive = Vector3d(1.0, 0.5, 0.5) * 0;
//	spheres[4].refl_coeff = 0.00;
//
//	*scene = spheres;
//	*nSpheres = 7;
//}

//__global__ void setup_scene_kernel(Sphere **scene, int *nSpheres)
//{
//	Sphere *spheres = new Sphere[16];
//
//	for(int i = 0; i < 15; i += 4)
//	{
//		spheres[i].radius = 1.5;
//		spheres[i].position.x() = -10;
//		spheres[i].position.y() = 1.5;
//		spheres[i].position.z() = i * -10;
//		spheres[i].diffuse.x() = 1;
//		spheres[i].diffuse.y() = 1;
//		spheres[i].diffuse.z() = 1;
//		spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//		spheres[i].refl_coeff = 0.0;
//
//		spheres[i+1].radius = 1.5;
//		spheres[i+1].position.x() = 10;
//		spheres[i+1].position.y() = 1.5;
//		spheres[i+1].position.z() = i * -10;
//		spheres[i+1].diffuse.x() = 1;
//		spheres[i+1].diffuse.y() = 1;
//		spheres[i+1].diffuse.z() = 1;
//		spheres[i+1].emissive = Vector3d(1.0, 1.0, 1.0) * 0;
//		spheres[i+1].refl_coeff = 0.0;
//
//		spheres[i+2].radius = 0.5;
//		spheres[i+2].position.x() = -10 + 2;
//		spheres[i+2].position.y() = 1.5;
//		spheres[i+2].position.z() = i * -10;
//		spheres[i+2].diffuse.x() = 1;
//		spheres[i+2].diffuse.y() = 1;
//		spheres[i+2].diffuse.z() = 1;
//		spheres[i+2].emissive = Vector3d(1.0, 0.1, 0.1) * 4;
//		spheres[i+2].refl_coeff = 0.0;
//	}
//
//	spheres[16].radius = 8000;
//	spheres[16].position.x() = 0;
//	spheres[16].position.y() = -8000;
//	spheres[16].position.z() = 0;
//	spheres[16].diffuse.x() = 1;
//	spheres[16].diffuse.y() = 1;
//	spheres[16].diffuse.z() = 1;
//	spheres[16].emissive = Vector3d(1.0, 0.5, 0.5) * 0;
//	spheres[16].refl_coeff = 0.00;
//
//	*scene = spheres;
//	*nSpheres = 17;
//}




__global__ void setup_scene_kernel(Sphere **scene, int *nSpheres)
{
	Sphere *spheres = new Sphere[8];

	spheres[0].radius = 400;
	spheres[0].position.x() = 0;
	spheres[0].position.y() = -400;
	spheres[0].position.z() = 0;
	spheres[0].diffuse.x() = 1;
	spheres[0].diffuse.y() = 1;
	spheres[0].diffuse.z() = 1;
	spheres[0].emissive = Vector3d(1.0, 1.0, 1.0) * 0.00;
	spheres[0].refl_coeff = 0;

	spheres[1].radius = 0.01;
	spheres[1].position.x() = 0;
	spheres[1].position.y() = 2.0;
	spheres[1].position.z() = 0;
	spheres[1].diffuse.x() = 1;
	spheres[1].diffuse.y() = 0;
	spheres[1].diffuse.z() = 0;
	spheres[1].emissive = Vector3d(0.5, 0.5, 0.5) * 10;
	spheres[1].refl_coeff = 0.0;

	double angle_step = (3.14159265359 * 2) / 8.0;
	for(int i = 2; i < 10; ++i)
	{
		double angle = angle_step * (i-2);
		double x = sin(angle) * 3;
		double z = cos(angle) * 3;
		double y = 0.5 + (0.3 * i);

		spheres[i].radius = 0.5;
		spheres[i].position.x() = x;
		spheres[i].position.y() = 0.5;
		spheres[i].position.z() = z;
		spheres[i].diffuse.x() = 1.0;
		spheres[i].diffuse.y() = 1.0;
		spheres[i].diffuse.z() = 1.0;
		spheres[i].emissive = Vector3d(0.0, 0.0, 0.0);
		spheres[i].refl_coeff = 0.0;
	}

	int i = 10;
	spheres[i].radius = 400;
	spheres[i].position.x() = 405;
	spheres[i].position.y() = 0;
	spheres[i].position.z() = 0;
	spheres[i].diffuse.x() = 0.6;
	spheres[i].diffuse.y() = 0.6;
	spheres[i].diffuse.z() = 1;
	spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0.00;
	spheres[i].refl_coeff = 0;

	i++;
	spheres[i].radius = 400;
	spheres[i].position.x() = -405;
	spheres[i].position.y() = 0;
	spheres[i].position.z() = 0;
	spheres[i].diffuse.x() = 1;
	spheres[i].diffuse.y() = 0.6;
	spheres[i].diffuse.z() = 0.6;
	spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0.00;
	spheres[i].refl_coeff = 0;

	i++;
	spheres[i].radius = 400;
	spheres[i].position.x() = 0;
	spheres[i].position.y() = 0;
	spheres[i].position.z() = -405;
	spheres[i].diffuse.x() = 0.6;
	spheres[i].diffuse.y() = 1;
	spheres[i].diffuse.z() = 0.6;
	spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0.00;
	spheres[i].refl_coeff = 0;

	i++;
	spheres[i].radius = 400;
	spheres[i].position.x() = 0;
	spheres[i].position.y() = 405;
	spheres[i].position.z() = 0;
	spheres[i].diffuse.x() = 0.6;
	spheres[i].diffuse.y() = 1;
	spheres[i].diffuse.z() = 0.6;
	spheres[i].emissive = Vector3d(1.0, 1.0, 1.0) * 0.00;
	spheres[i].refl_coeff = 0;

	for(int j = 1; j <= 8; ++j)
	{
		double angle = angle_step * (i-2);
		double x = sin(angle) * 3;
		double z = cos(angle) * 3;
		double y = 0.5 + (0.3 * i);

		spheres[i + j].radius = 0.2;
		spheres[i + j].position.x() = x;
		spheres[i + j].position.y() = 0.5 + y;
		spheres[i + j].position.z() = z;
		spheres[i + j].diffuse.x() = 1.0;
		spheres[i + j].diffuse.y() = 1.0;
		spheres[i + j].diffuse.z() = 1.0;
		spheres[i + j].emissive = Vector3d(0.0, 0.0, 0.0);
		spheres[i + j].refl_coeff = 1.0;
	}


	*scene = spheres;
	*nSpheres = 22;
}

cudaError_t setup_scene(Sphere **spheres, int *nSpheres)
{
	setup_scene_kernel<<<1, 1>>>(spheres, nSpheres);

	return cudaDeviceSynchronize();
}

cudaError_t setup_octree(Sphere **spheres, int nspheres, Octree **octree)
{
	setup_octree_kernel<<<1, 1>>>(spheres, nspheres, octree);

	return cudaDeviceSynchronize();
}
