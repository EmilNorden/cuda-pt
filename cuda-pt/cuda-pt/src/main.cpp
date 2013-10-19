#include <GL/glew.h>
#include "sdl.h"
#include "SDL_syswm.h"
#include "window.h"
//
//#include "cudaGL.h"
#include "cuda_gl_interop.h"

#include <iostream>

cudaError_t dostuff(void *ptr);

#define CUDA_CALL(x) if(x != cudaSuccess) { exit(1); }

#define GL_CALL(x) if(x != GL_NO_ERROR) { exit(1); } 


int main(int argc, char **argv)
{
	SDL_SysWMinfo info;

	SDL_Init(SDL_INIT_EVERYTHING);

	SDLWindow window(SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, "CUDA Pathtracing");
	SDL_VERSION(&info.version);
	
	 if(!SDL_GetWindowWMInfo(window.get(),&info))
	 {
		 std::cout << "SDL_GetWindowWMInfo failed.\n";
		 exit(1);
	 }

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
	
	CUDA_CALL(cudaGLSetGLDevice(0))
	
	GLuint bufferID;
	// Generate a buffer ID
	glGenBuffers(1,&bufferID); 
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	// Allocate data for the buffer
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 800 * 600 * 4, 		NULL, GL_DYNAMIC_COPY);	CUDA_CALL(cudaGLRegisterBufferObject(bufferID))	GLuint textureID;	// Enable Texturing
	glEnable(GL_TEXTURE_2D);
	// Generate a texture ID
	glGenTextures(1,&textureID); 
	// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID); 
	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it 
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_BGRA, 
	GL_UNSIGNED_BYTE, NULL);
	// Must set the filter mode, GL_LINEAR enables interpolation when scaling 
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	GL_CALL(glGetError())	void *cudaMemory;

	


	bool running = true;
	while(running)
	{
		SDL_Event e;
		while(SDL_PollEvent(&e))
		{
			if(e.type == SDL_QUIT)
				running = false;
		}



		CUDA_CALL(cudaGLMapBufferObject(&cudaMemory, bufferID))

		dostuff(cudaMemory);

		CUDA_CALL(cudaGLUnmapBufferObject(bufferID))

		// Select the appropriate buffer 
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID); 
		// Select the appropriate texture
		glBindTexture( GL_TEXTURE_2D, textureID);
		// Make a texture from the buffer
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 800, 600,
		GL_BGRA, GL_UNSIGNED_BYTE, NULL);		GL_CALL(glGetError())		glBegin(GL_QUADS);
			glTexCoord2f( 0, 1.0f); 
			glVertex3f(0,0,0);
			glTexCoord2f(0,0);
			glVertex3f(0,1.0f,0);
			glTexCoord2f(1.0f,0);
			glVertex3f(1.0f,1.0f,0);
			glTexCoord2f(1.0f,1.0f);
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