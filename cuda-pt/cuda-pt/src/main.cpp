#include <GL/glew.h>
#include "sdl.h"
#include "SDL_syswm.h"
#include "window.h"
//
//#include "cudaGL.h"
#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "curand_kernel.h"
#include "cuda_gl_interop.h"
#include "stack.h"

#include <iostream>

#include "camera.h"
#include "gametime.h"
#include "mouse_device.h"

cudaError_t dostuff(void *ptr, Camera *device_camera, int seed, Vector3d *light_d, int sample, Vector3d *buffer_d, curandState *state);
cudaError_t clearBuffer(Vector3d *ptr);
cudaError_t init_curand(curandState *rand_state_d, unsigned long *seeds);

#define PI 3.14159265359
//#define CUDA_CALL(x) if(x != cudaSuccess) { exit(1); }
//#define GL_CALL(x) if(x != GL_NO_ERROR) { exit(1); } 

#define RESOLUTION_WIDTH	800
#define RESOLUTION_HEIGHT	600


int main(int argc, char **argv)
{
	SDL_SysWMinfo info;

	std::cout << "Initializing SDL\n";
	SDL_Init(SDL_INIT_EVERYTHING);

	SDLWindow window(SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, "CUDA Pathtracing");
	SDL_VERSION(&info.version);
	
	 if(!SDL_GetWindowWMInfo(window.get(),&info))
	 {
		 std::cout << "SDL_GetWindowWMInfo failed.\n";
		 exit(1);
	 }

	 std::cout << "Initializing OpenGL...\n";
	static PIXELFORMATDESCRIPTOR pfd2;
	
	pfd2.nSize = sizeof(PIXELFORMATDESCRIPTOR);

	static PIXELFORMATDESCRIPTOR pfd=
	{
		sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
		1, // Version Number
		PFD_DRAW_TO_WINDOW | // Format Must Support Window
		PFD_SUPPORT_OPENGL | // Format Must Support OpenGL
		PFD_DOUBLEBUFFER, // Must Support Double Buffering
		PFD_TYPE_RGBA, // Request An RGBA Format
		8, // Select Our Color Depth, 8 bits / channel
		0, 0, 0, 0, 0, 0, // Color Bits Ignored
		0, // No Alpha Buffer
		0, // Shift Bit Ignored
		0, // No Accumulation Buffer
		0, 0, 0, 0, // Accumulation Bits Ignored
		32, // 32 bit Z-Buffer (Depth Buffer) 
		0, // No Stencil Buffer
		0, // No Auxiliary Buffer
		PFD_MAIN_PLANE, // Main Drawing Layer
		0, // Reserved
		0, 0, 0 // Layer Masks Ignored
	};

	HWND hWnd = info.info.win.window;

	HDC hDc = GetDC(hWnd);

	GLuint pixel_format = ChoosePixelFormat(hDc, &pfd);

	SetPixelFormat(hDc, pixel_format, &pfd);

	HGLRC glc = wglCreateContext(hDc);
	wglMakeCurrent(hDc, glc);
	
	GL_CALL(glewInit())

	glViewport(0, 0, 800, 600);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glEnable(GL_DEPTH_TEST); // ?? Should this not be disabled?

	glClearColor(1.0f, 0.0f, 1.0f, 1.0f);

	GL_CALL(glGetError())
	
	std::cout << "Initializing CUDA...\n";
	CUDA_CALL(cudaGLSetGLDevice(0))
	
	GLuint bufferID;
	// Generate a buffer ID
	glGenBuffers(1,&bufferID); 
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	// Allocate data for the buffer
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 800 * 600 * 4 * 4, 
		NULL, GL_DYNAMIC_COPY);


	CUDA_CALL(cudaGLRegisterBufferObject(bufferID))

	GLuint textureID;

	// Enable Texturing
	glEnable(GL_TEXTURE_2D);
	// Generate a texture ID
	glGenTextures(1,&textureID); 
	// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID); 
	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it 
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	GLenum enu = glGetError();
	GL_CALL(enu);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, 800, 600, 0, GL_RGBA, GL_UNSIGNED_INT, NULL);
	//glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, 800, 600, 0, GL_RGB, GL_FLOAT, NULL);
	enu = glGetError();
	GL_CALL(enu);
	// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

	enu = glGetError();
	GL_CALL(enu)

	void *cudaMemory;
	Vector3d cam_pos(0, 5, 20);
	Vector3d cam_target = Vector3d(0, 0, 0) - cam_pos;
	cam_target.normalize();

	Vector3d focus_point(0, 1.5, 5);
	double focal_length = (focus_point - cam_pos).length();
	Camera c(cam_pos, cam_target, Vector3d(0, 1, 0), PI / 4.0, 800.0 / 600.0, Vector2i(800, 600), focal_length);

	Camera *device_c; 
	CUDA_CALL(cudaMalloc(&device_c, sizeof(Camera)));

	/* Set CUDA Data Stack size */
	std::cout << "Setting CUDA Data stack size...\n";
	size_t limit;
	CUDA_DRIVER_CALL(cuCtxGetLimit(&limit, CU_LIMIT_STACK_SIZE))
	limit *= 10;
	CUDA_DRIVER_CALL(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, limit));
	std::cout << "Successfully set CUDA Data stack size to " << limit << "\n";

	Vector3d light_h (0, 10, 00);

	Vector3d *light_d;
	CUDA_CALL(cudaMalloc(&light_d, sizeof(Vector3d)));

	Vector3d *buffer_d;
	CUDA_CALL(cudaMalloc(&buffer_d, sizeof(Vector3d) * 800 * 600));
	cudaMemset(buffer_d, 0, sizeof(Vector3d) * 800 * 600);

	std::cout << "Initializing curand...\n";
	/* Set up CURAND */
	curandState *rand_state_d;
	CUDA_CALL(cudaMalloc(&rand_state_d, sizeof(curandState) * 800 * 600));
	unsigned long *seeds_h = new unsigned long[800 * 600];
	unsigned long *seeds_d;
	for(int i = 0; i < 800 * 600; ++i)
	{
		seeds_h[i] = SDL_GetTicks() + i;
	}

	CUDA_CALL(cudaMalloc(&seeds_d, sizeof(unsigned long) * 800 * 600));
	CUDA_CALL(cudaMemcpy(seeds_d, seeds_h, sizeof(unsigned long) * 800 * 600,  cudaMemcpyHostToDevice));
	delete[] seeds_h;

	CUDA_CALL(init_curand(rand_state_d, seeds_d));

	MouseDevice mouse;
	GameTime timer(SDL_GetTicks());
	bool frame_changed = false;
	int sample = 0;
	bool running = true;
	std::cout << "Done! Entering render loop\n";
	bool dragging = false;
	double rot_x = PI;
	double rot_y = PI;
	Vector2i drag_coord;
	while(running)
	{
		mouse.frame_update();
		timer.frameUpdate(SDL_GetTicks());

		Vector3d movement(0, 0, 0);
		SDL_Event e;
		while(SDL_PollEvent(&e))
		{
			if(e.type == SDL_QUIT)
				running = false;
		}
		int numKeys;
		const Uint8 *data = SDL_GetKeyboardState(&numKeys);
		if(data[SDL_SCANCODE_LEFT])
		{
			frame_changed = true;
			movement.x()--;//x--;
		}
		if(data[SDL_SCANCODE_RIGHT])
		{
			frame_changed = true;
			movement.x()++; //x++;
		}
		if(data[SDL_SCANCODE_UP])
		{
			frame_changed = true;
			movement.y()++;//y++;
		}
		if(data[SDL_SCANCODE_DOWN])
		{
			frame_changed = true;
			movement.y()--;//y--;
		}
		if(data[SDL_SCANCODE_W])
		{
			frame_changed = true;
			c.set_position(c.position() + c.direction());
		}
		if(data[SDL_SCANCODE_S])
		{
			frame_changed = true;
			c.set_position(c.position() - c.direction());
		}
		if(data[SDL_SCANCODE_ESCAPE])
		{
			running = false;
		}

		if(mouse.left_button() & Pressed && mouse.left_button() & ChangedThisFrame)
		{
			dragging = true;
			drag_coord.x() = mouse.x();
			drag_coord.y() = mouse.y();
			std::cout << "Begin drag!\n";
		}
		else if(mouse.left_button() & Released)
		{
			dragging = false;
		}

		if(dragging)
		{
			frame_changed = true;
			Vector2i mouse_pos(mouse.x(), mouse.y());
			Vector2i diff = mouse_pos - drag_coord;
			drag_coord = mouse_pos;

			rot_x += diff.x() * 0.005;
			rot_y += diff.y() * 0.005;

			std::cout << "Rotation: " << rot_x << ", " << rot_y << "\n";
			
			Vector3d direction = Vector3d(std::sin(rot_x), 0, std::cos(rot_x)) + Vector3d(0, std::sin(rot_y), std::cos(rot_y));
			direction.normalize();
			c.set_direction(direction);
		}



		

		if(frame_changed)
		{
			movement.multiply(timer.frameTime() * 0.01);

			//c.set_position(c.position() + (c.direction() * movement));
			light_h += movement;
			frame_changed = false;
			sample = 0;
			//clearBuffer(buffer_d);
		}


		CUDA_CALL(cudaGLMapBufferObject(&cudaMemory, bufferID));

		//c.set_position(Vector3d(x, y, 0));
		c.update();
		//c.set_focal_length((light_h - c.position()).length());
		CUDA_CALL(cudaMemcpy(device_c, &c, sizeof(Camera), cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(light_d, &light_h, sizeof(Vector3d), cudaMemcpyHostToDevice));

		
		dostuff(cudaMemory, device_c, static_cast<int>(SDL_GetTicks()), light_d, sample++, buffer_d, rand_state_d);

		CUDA_CALL(cudaGLUnmapBufferObject(bufferID))

		// Select the appropriate buffer 
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID); 
		// Select the appropriate texture
		glBindTexture( GL_TEXTURE_2D, textureID);
		// Make a texture from the buffer
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 800, 600,
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

		GL_CALL(glGetError())
		
		SwapBuffers(hDc);
	}

	CUDA_CALL(cudaGLUnregisterBufferObject(bufferID))

	glDeleteBuffers(1, &bufferID);

	SDL_Quit();
	return 0;
}