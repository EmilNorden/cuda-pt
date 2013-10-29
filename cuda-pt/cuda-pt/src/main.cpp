#include <GL/glew.h>
#include "sdl.h"
#include "SDL_ttf.h"
#include "window.h"

#include "error_assertion.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "curand_kernel.h"
#include "cuda_gl_interop.h"
#include "stack.h"

#include <iostream>
#include <chrono>
#include <random>

#include "sphere.h"
#include "camera.h"
#include "gametime.h"
#include "mouse_device.h"
#include "shared_device_pointer.h"
#include "font.h"
#include "texture.h"
#include "cmd_args.h"

cudaError_t setup_scene(Sphere **scene, int *nSpheres);
cudaError_t focus_camera(Camera *device_camera, const Vector2i *focus_point_d, Sphere **scene, int nSpheres);
cudaError_t ray_trace(void *ptr, Camera *device_camera, int sample, Vector3d *buffer_d, curandState *state, const Vector2i &resolution_h, const Vector2i *resolution, Sphere **scene, int nSpheres);
cudaError_t clearBuffer(Vector3d *ptr);
cudaError_t init_curand(curandState *rand_state_d, unsigned long *seeds, const Vector2i &resolution_h, const Vector2i *resolution);

#define PI 3.14159265359

#define RESOLUTION_WIDTH	1280
#define RESOLUTION_HEIGHT	720
//#define RESOLUTION_WIDTH	800
//#define RESOLUTION_HEIGHT	600

Vector2i resolution_h(RESOLUTION_WIDTH, RESOLUTION_HEIGHT);

TTF_Font	*font;

GLuint	bufferID;
GLuint	textureID;
int		nSpheres;

// Device memory pointers
Camera		*camera_d; 
Vector3d	*accu_buffer_d;
curandState *rand_state_d;
Vector2i	*resolution_d;
Vector2i	*focus_point_d;
Sphere		**scene_d;


SDL_GLContext init_gl(SDLWindow &window)
{
	SDL_GLContext context = SDL_GL_CreateContext(window.get());
	
	GL_CALL(glewInit())

	glViewport(0, 0, RESOLUTION_WIDTH, RESOLUTION_HEIGHT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);

	GL_CALL(glGetError());

	// Generate a buffer ID
	glGenBuffers(1,&bufferID); 
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	// Allocate data for the buffer
	glBufferData(GL_PIXEL_UNPACK_BUFFER, RESOLUTION_WIDTH * RESOLUTION_HEIGHT * 4 * 4, 
		NULL, GL_DYNAMIC_COPY);

	// Enable alpha blend
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	// Enable Texturing
	glEnable(GL_TEXTURE_2D);
	// Generate a texture ID
	glGenTextures(1,&textureID); 
	// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID); 
	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it 
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	
	// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	GL_CALL(glGetError());

	return context;
}

void print_device_properties(const cudaDeviceProp &prop)
{
	std::cout << "Name:\t\t\t" << prop.name << "\n";
	std::cout << "Compute Capability:\t" << prop.major << "." << prop.minor << "\n";
	std::cout << "Global memory:\t\t" << prop.totalGlobalMem / (1024.0 * 1024.0) << "Mb\n";
	std::cout << "Clockrate:\t\t" << prop.clockRate << "\n";
	std::cout << "Memory clockrate:\t" << prop.memoryClockRate << "\n";
}

void choose_cuda_device()
{
	int device_count;
	CUDA_CALL(cudaGetDeviceCount(&device_count));
	
	if(device_count == 0)
	{
		std::cout << "No CUDA capable devices found!\n";
		exit(EXIT_FAILURE);
	}

	std::cout << "======\n";
	std::cout << "Found " << device_count << " CUDA capable devices\n";

	int device_id = 0;

	std::cout << "Choosing device " << device_id << ":\n";
	cudaSetDevice(device_id);
	cudaDeviceProp device_prop;
	CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_id));
	print_device_properties(device_prop);
	std::cout << "======\n";
}

void init_cuda()
{
	choose_cuda_device();

	CUDA_CALL(cudaGLSetGLDevice(0))
	CUDA_CALL(cudaGLRegisterBufferObject(bufferID))

	/* Set CUDA Data Stack size */
	std::cout << "Setting CUDA Data stack size...\n";
	size_t stack_size;
	CUDA_DRIVER_CALL(cuCtxGetLimit(&stack_size, CU_LIMIT_STACK_SIZE))
	stack_size *= 3;
	CUDA_DRIVER_CALL(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_size));
	std::cout << "Successfully set CUDA Data stack size to " << stack_size << " bytes\n";

	CUDA_CALL(cudaMalloc(&camera_d, sizeof(Camera)));

	CUDA_CALL(cudaMalloc(&accu_buffer_d, sizeof(Vector3d) * RESOLUTION_WIDTH * RESOLUTION_HEIGHT));
	CUDA_CALL(cudaMemset(accu_buffer_d, 0, sizeof(Vector3d) * RESOLUTION_WIDTH * RESOLUTION_HEIGHT));

	CUDA_CALL(cudaMalloc(&resolution_d, sizeof(Vector2i)));
	CUDA_CALL(cudaMemcpy(resolution_d, &resolution_h, sizeof(Vector2i), cudaMemcpyHostToDevice));
}

void init_curand()
{
	CUDA_CALL(cudaMalloc(&rand_state_d, sizeof(curandState) * RESOLUTION_WIDTH * RESOLUTION_HEIGHT));
	
	unsigned long *seeds_d;

	// Randomly generate seeds for the kernels curand states.
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::minstd_rand0 generator (seed);

	unsigned long *seeds_h = new unsigned long[RESOLUTION_WIDTH * RESOLUTION_HEIGHT];
	for(int i = 0; i < RESOLUTION_WIDTH * RESOLUTION_HEIGHT; ++i)
	{
		seeds_h[i] = generator();
	}

	CUDA_CALL(cudaMalloc(&seeds_d, sizeof(unsigned long) * RESOLUTION_WIDTH * RESOLUTION_HEIGHT));
	CUDA_CALL(cudaMemcpy(seeds_d, seeds_h, sizeof(unsigned long) * RESOLUTION_WIDTH * RESOLUTION_HEIGHT,  cudaMemcpyHostToDevice));
	delete[] seeds_h;

	CUDA_CALL(init_curand(rand_state_d, seeds_d, Vector2i(RESOLUTION_WIDTH, RESOLUTION_HEIGHT), resolution_d));
}

void init_scene()
{
	int *nSpheres_d;
	cudaMalloc(&scene_d, sizeof(Sphere*));
	cudaMalloc(&nSpheres_d, sizeof(int));

	CUDA_CALL(setup_scene(scene_d, nSpheres_d));
	CUDA_CALL(cudaMemcpy(&nSpheres, nSpheres_d, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(nSpheres_d));
}

std::shared_ptr<Texture> RenderText(std::string message, SDLFont &font, 
                        SDL_Color color)
{
    SDL_Surface *surf = TTF_RenderText_Blended(font.get(), message.c_str(), color);
	std::shared_ptr<Texture> tex(new Texture(surf));
	SDL_FreeSurface(surf);

	return tex;
}

int main(int argc, char **argv)
{
	CmdArgs args(argc, argv);
	std::cout << "Initializing SDL\n";
	SDL_Init(SDL_INIT_EVERYTHING);
	TTF_Init();
	SDLWindow window(SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, false, "CUDA Pathtracing");

	std::cout << "Initializing OpenGL...\n";
	SDL_GLContext context = init_gl(window);

	std::cout << "Initializing CUDA...\n";
	init_cuda();

	std::cout << "Initializing curand...\n";
	/* Set up CURAND */
	
	init_curand();

	std::cout << "Initializing scene...\n";
	init_scene();
	// Camera setup
	Vector3d cam_pos(0, 5, 20);
	Vector3d cam_target = Vector3d(0, 0, 0) - cam_pos;
	cam_target.normalize();

	Vector3d focus_point(0, 1.5, 5);
	double focal_length = (focus_point - cam_pos).length();
	Camera camera_h(cam_pos, cam_target, Vector3d(0, 1, 0), PI / 4.0, (double)RESOLUTION_WIDTH / RESOLUTION_HEIGHT, Vector2i(RESOLUTION_WIDTH, RESOLUTION_HEIGHT), focal_length);

	
	
	void *pixel_buffer_object_d;
	MouseDevice mouse;
	GameTime timer(SDL_GetTicks());
	bool frame_changed = false;
	int sample = 0;
	bool running = true;
	
	bool dragging = false;
	double view_rot_x = PI;
	double view_rot_y = PI;

	Vector2i prev_mouse_coord;
	
	CUDA_CALL(cudaMalloc(&focus_point_d, sizeof(Vector2i)));

	uint32_t sample_start_time = timer.realTime();
	std::cout << "Done! Entering render loop\n";
	while(running)
	{
		mouse.frame_update();
		timer.frame_update(SDL_GetTicks());
		std::cout << timer.frameTime() << "ms\n";
		SDL_Event e;
		while(SDL_PollEvent(&e))
		{
			if(e.type == SDL_QUIT)
				running = false;
		}
		int numKeys;
		const Uint8 *data = SDL_GetKeyboardState(&numKeys);
		if(data[SDL_SCANCODE_W])
		{
			frame_changed = true;
			camera_h.set_position(camera_h.position() + camera_h.direction());
		}
		if(data[SDL_SCANCODE_S])
		{
			frame_changed = true;
			camera_h.set_position(camera_h.position() - camera_h.direction());
		}
		if(data[SDL_SCANCODE_ESCAPE])
		{
			running = false;
		}
		if(data[SDL_SCANCODE_TAB])
		{
			double duration = (timer.realTime() - sample_start_time) / 1000.0f;
			uint32_t rendered_data = (RESOLUTION_WIDTH * RESOLUTION_HEIGHT * 4 * sample) / duration;
			std::cout << "Current sample count: " << sample << ". Render bandwidth: " << rendered_data / (1024*1024) << " Mb/s\n";
		}

		if(mouse.left_button() & Pressed && mouse.left_button() & ChangedThisFrame)
		{
			dragging = true;
			prev_mouse_coord.x() = mouse.x();
			prev_mouse_coord.y() = mouse.y();
		}
		else if(mouse.left_button() & Released)
		{
			dragging = false;
		}

		if(dragging)
		{
			frame_changed = true;
			Vector2i mouse_pos(mouse.x(), mouse.y());
			Vector2i diff = mouse_pos - prev_mouse_coord;
			prev_mouse_coord = mouse_pos;

			view_rot_x += diff.x() * 0.005;
			view_rot_y += diff.y() * 0.005;

			Vector3d direction = Vector3d(std::sin(view_rot_x), 0, std::cos(view_rot_x)) + Vector3d(0, std::sin(view_rot_y), std::cos(view_rot_y));
			direction.normalize();
			camera_h.set_direction(direction);
		}

		if(frame_changed)
		{
			// If anything has changed this frame,
			// we need to set the sample count to 0 so
			// that the pixex buffer object is completely overwritten with new data
			frame_changed = false;
			sample = 0;
			sample_start_time = timer.realTime();
			camera_h.update();
			CUDA_CALL(cudaMemcpy(camera_d, &camera_h, sizeof(Camera), cudaMemcpyHostToDevice));
		}

		if(mouse.right_button() & Pressed && mouse.right_button() & ChangedThisFrame)
		{
			int mouse_x = mouse.x();
			int mouse_y = mouse.y();
			Vector2i window_size = window.get_size();

			double normalized_x = mouse_x / (double)window_size.x();
			double normalized_y = 1 - (mouse_y / (double)window_size.y());

			Vector2i focus_point(static_cast<int>(RESOLUTION_WIDTH * normalized_x), static_cast<int>(RESOLUTION_HEIGHT * normalized_y));
			CUDA_CALL(cudaMemcpy(focus_point_d, &focus_point, sizeof(Vector2i), cudaMemcpyHostToDevice));

			focus_camera(camera_d, focus_point_d, scene_d, nSpheres);	
			sample = 0;
		}

		CUDA_CALL(cudaGLMapBufferObject(&pixel_buffer_object_d, bufferID));

		ray_trace(pixel_buffer_object_d, camera_d, sample++, accu_buffer_d, rand_state_d, resolution_h, resolution_d, scene_d, nSpheres);

		CUDA_CALL(cudaGLUnmapBufferObject(bufferID))

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID); 
		glBindTexture( GL_TEXTURE_2D, textureID);
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, RESOLUTION_WIDTH, RESOLUTION_HEIGHT,
		GL_BGRA, GL_UNSIGNED_BYTE, NULL);

		GL_CALL(glGetError());

		glBegin(GL_QUADS);
			glTexCoord2f( 0, 0.0f); 
			glVertex3f(0,0,0);
			glTexCoord2f(0,1.0f);
			glVertex3f(0,1.0f,0);
			glTexCoord2f(1.0f, 1.0f);
			glVertex3f(1.0f,1.0f,0);
			glTexCoord2f(1.0f, 0.0f);
			glVertex3f(1.0f,0,0);
		glEnd();

		/*glBindTexture( GL_TEXTURE_2D, texture->glTexture() );

		glBegin(GL_QUADS);
			glTexCoord2f(0.0f, 0.0f); 
			glVertex3f(0,0,0);

			glTexCoord2f(0,1.0f);
			glVertex3f(0,1.0f,0);

			glTexCoord2f(1.0f, 1.0f);
			glVertex3f(1.0f,1.0f,0);

			glTexCoord2f(1.0f, 0.0f);
			glVertex3f(1.0f,0,0);
		glEnd();*/

		GL_CALL(glGetError());
		
		SDL_GL_SwapWindow(window.get());
	}

	// CUDA cleanup
	CUDA_CALL(cudaGLUnregisterBufferObject(bufferID));
	CUDA_CALL(cudaDeviceReset());

	cudaFree(camera_d);
	cudaFree(accu_buffer_d);
	cudaFree(resolution_d);
	cudaFree(focus_point_d);

	// GL cleanup
	glDeleteBuffers(1, &bufferID);
	glDeleteTextures(1, &textureID);

	// SDL cleanup
	SDL_GL_DeleteContext(context);

	SDL_Quit();
	return 0;
}