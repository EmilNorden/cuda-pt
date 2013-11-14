#include "GL/glew.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "sdl.h"
#include "SDL_ttf.h"
#include "window.h"
#include "error_assertion.h"
#include "camera.h"
#include "gametime.h"
#include "mouse_device.h"
#include "font.h"
#include "texture.h"
#include "cmd_args.h"
#include "opengl_surface.h"
#include "cuda_raytracer.h"

#include <iostream>

cudaError_t setup_scene(Sphere **scene, int *nSpheres);

#define PI 3.14159265359

#define RESOLUTION_WIDTH	640
#define RESOLUTION_HEIGHT	480

TTF_Font *font;

int	nSpheres;

// Device memory pointers
Sphere		**scene_d;

void cleanup_cuda()
{
	CUDA_CALL(cudaDeviceReset());
}

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

	// Enable Texturing
	glEnable(GL_TEXTURE_2D);
 
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
	atexit(cleanup_cuda);
	choose_cuda_device();

	CUDA_CALL(cudaGLSetGLDevice(0));

	// As far as I understand, the cuda context must be created before calling any cuda driver API function.
	// So before setting the stack size below, I "have" to call something, thats why this cudaDeviceSynchronize call is here.
	cudaDeviceSynchronize();

	/* Set CUDA Data Stack size */
	std::cout << "Setting CUDA Data stack size...\n";
	size_t stack_size;
	CUDA_DRIVER_CALL(cuCtxGetLimit(&stack_size, CU_LIMIT_STACK_SIZE))
	stack_size *= 3;
	CUDA_DRIVER_CALL(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_size));
	std::cout << "Successfully set CUDA Data stack size to " << stack_size << " bytes\n";
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
	std::cout << "Initializing SDL...\n";
	SDL_Init(SDL_INIT_EVERYTHING);
	TTF_Init();
	SDLWindow window(SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, false, "CUDA Pathtracing");

	std::cout << "Initializing OpenGL...\n";
	SDL_GLContext context = init_gl(window);

	std::cout << "Initializing CUDA...\n";
	init_cuda();

	std::cout << "Initializing scene...\n";
	init_scene();

	CudaRayTracer tracer;

	// Camera setup
	Vector3d cam_pos(0, 5, 20);
	Vector3d cam_target = Vector3d(0, 0, 0) - cam_pos;
	cam_target.normalize();

	Vector3d focus_point(0, 1.5, 5);
	double focal_length = (focus_point - cam_pos).length();
	Camera camera_h(cam_pos, cam_target, Vector3d(0, 1, 0), PI / 4.0, (double)RESOLUTION_WIDTH / RESOLUTION_HEIGHT, Vector2i(RESOLUTION_WIDTH, RESOLUTION_HEIGHT), focal_length);

	auto surface = OpenGLSurface::create(RESOLUTION_WIDTH, RESOLUTION_HEIGHT);
	tracer.set_surface(surface);
	
	MouseDevice mouse;
	GameTime timer(SDL_GetTicks());
	bool frame_changed = false;
	
	
	bool dragging = false;
	double view_rot_x = PI;
	double view_rot_y = PI;

	Vector2i prev_mouse_coord;
	camera_h.update();
	uint32_t sample_start_time = timer.realTime();
	std::cout << "Entering render loop\n";
	bool running = true;
	while(running)
	{
		mouse.frame_update();
		timer.frame_update(SDL_GetTicks());
		std::cout << timer.frameTime() << "ms    \r";
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
		if(data[SDL_SCANCODE_UP])
		{
			frame_changed = true;
			camera_h.set_focal_length(camera_h.focal_length() + 0.1);
		}
		if(data[SDL_SCANCODE_DOWN])
		{
			frame_changed = true;
			camera_h.set_focal_length(camera_h.focal_length() - 0.1);	
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

		if(mouse.right_button() & Released &&
			mouse.right_button() & ChangedThisFrame)
		{
			Vector2i mouse_pos(mouse.x(), mouse.y());
			double camera_dist;
			CUDA_CALL(tracer.get_camera_distance(camera_h, mouse_pos, scene_d, nSpheres, camera_dist));
			camera_h.set_focal_length(camera_dist);
			frame_changed = true;
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
			frame_changed = false;
			camera_h.update();
		}

		tracer.render(camera_h, scene_d, nSpheres);
		
		surface->draw();

		GL_CALL(glGetError());
		
		SDL_GL_SwapWindow(window.get());

		camera_h.reset_update_flag();
	}

	// SDL cleanup
	SDL_GL_DeleteContext(context);

	SDL_Quit();
	return 0;
}