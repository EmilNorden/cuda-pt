#include "GL/glew.h"
#include "GL/wglext.h"
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
#include "sprite.h"
#include "octree.h"

#include <iostream>
#include <sstream>
#include <ctime>


cudaError_t setup_scene(Sphere **scene, int *nSpheres);
cudaError_t setup_octree(Sphere **spheres, int nspheres, Octree **octree);

#define PI 3.14159265359

#define RESOLUTION_WIDTH	1920
#define RESOLUTION_HEIGHT	1080

TTF_Font *font;

int	nSpheres;

// Device memory pointers
Sphere		**scene_d;

Octree		**octree_d;

bool WGLExtensionSupported(const char *extension_name)
{
    // this is pointer to function which returns pointer to string with list of all wgl extensions
    PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglGetExtensionsStringEXT = NULL;

    // determine pointer to wglGetExtensionsStringEXT function
    _wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC) wglGetProcAddress("wglGetExtensionsStringEXT");

    if (strstr(_wglGetExtensionsStringEXT(), extension_name) == NULL)
    {
        // string was not found
        return false;
    }

    // extension is supported
    return true;
}

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
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
 
	GL_CALL(glGetError());

#ifdef _WIN32
	if(WGLExtensionSupported("WGL_EXT_swap_control"))
	{
		std::cout << "WGL_EXT_swap_control supported. Disabling vsync.\n";
		PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(0);
	}
	else
	{
		std::cout << "WGL_EXT_swap_control not supported. Unable to disable vsync.\n";
	}
#endif

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
	//CUDA_DRIVER_CALL(cuCtxGetLimit(&stack_size, CU_LIMIT_STACK_SIZE))
	// Through very scientific methods (ie brute force testing)	 I have deduced my stack size to be 2705 bytes.
	stack_size = 2700;
	CUDA_DRIVER_CALL(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_size));
	std::cout << "Successfully set CUDA Data stack size to " << stack_size << " bytes\n";
}

void init_scene()
{
	int *nSpheres_d;
	cudaMalloc(&scene_d, sizeof(Sphere*));
	cudaMalloc(&nSpheres_d, sizeof(int));
	cudaMalloc(&octree_d, sizeof(Octree*));

	CUDA_CALL(setup_scene(scene_d, nSpheres_d));
	CUDA_CALL(cudaMemcpy(&nSpheres, nSpheres_d, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(nSpheres_d));

	//CUDA_CALL(setup_octree(scene_d, nSpheres, octree_d));
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
	SDLWindow window(SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, true, "CUDA Pathtracing");

	std::cout << "Initializing OpenGL...\n";
	SDL_GLContext context = init_gl(window);

	SDLFont sdl_font(std::string("../Resources/SourceSansPro-Regular.ttf"), 32);
	SDL_Color clr;
	clr.r = 255;
	clr.g = 0;
	clr.b = 0;
	clr.a = 255;

	srand(time(NULL));

	Sprite sprite_fps(RenderText("0 fps", sdl_font, clr), Vector2d(0, 0));
	//Sprite focal_fps(RenderText("Focal length: 0", sdl_font, clr), Vector2d(0, sprite_fps.texture()->height()));

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
	int frames = 0;
	uint32_t fps_timer = 0;
	while(running)
	{

		glClear(GL_COLOR_BUFFER_BIT);
		mouse.frame_update();
		timer.frame_update(SDL_GetTicks());
		
		fps_timer += timer.frameTime();
		if(fps_timer > 1000)
		{
			std::cout << "\r" << frames << " fps  ";
			std::stringstream ss;
			ss << "FPS: " << frames;
			GL_CALL(glGetError());
			//sprite_fps.set_texture(RenderText(ss.str(), sdl_font, clr));
			GL_CALL(glGetError());
			fps_timer -= 1000;
			frames = 0;
		}
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
			double dist;
			tracer.get_camera_distance(camera_h, Vector2i(RESOLUTION_WIDTH / 2, RESOLUTION_HEIGHT / 2), scene_d, nSpheres, dist);
			if(dist > 1000)
				dist = 1000;
			camera_h.set_focal_length(dist);
			frame_changed = false;
			camera_h.update();
		}

		tracer.render(camera_h, scene_d, nSpheres);
		
		surface->draw();

		
		
		//sprite_fps.draw(RESOLUTION_WIDTH, RESOLUTION_HEIGHT);

		glLoadIdentity();
		SDL_GL_SwapWindow(window.get());

		camera_h.reset_update_flag();
		frames++;
	}

	// SDL cleanup
	SDL_GL_DeleteContext(context);

	SDL_Quit();
	return 0;
}