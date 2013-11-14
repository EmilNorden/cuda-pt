#include <GL/glew.h>

#include "opengl_surface.h"
#include "opengl_surface.h"
#include "cuda_gl_interop.h"

#include "error_assertion.h"

OpenGLSurface::OpenGLSurface(size_t width, size_t height)
	: resolution_(width, height)
{
	glGenBuffers(1, &buffer_id_); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id_);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, 
		NULL, GL_DYNAMIC_COPY);

	glGenTextures(1, &texture_id_); 
	glBindTexture( GL_TEXTURE_2D, texture_id_); 
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, resolution_.x(), resolution_.y(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	GL_CALL(glGetError());

	CUDA_CALL(cudaGLRegisterBufferObject(buffer_id_));
	CUDA_CALL(cudaMalloc(&accu_buffer_d_, sizeof(Vector3d) * resolution_.x() * resolution_.y()));
	CUDA_CALL(cudaMemset(accu_buffer_d_, 0, sizeof(Vector3d) * resolution_.x() * resolution_.y()));

	CUDA_CALL(cudaMalloc(&resolution_d_, sizeof(Vector2i)));
	CUDA_CALL(cudaMemcpy(resolution_d_, &resolution_, sizeof(Vector2i), cudaMemcpyHostToDevice));
}

OpenGLSurface::~OpenGLSurface()
{
	CUDA_CALL(cudaFree(accu_buffer_d_));
	/* Commented out since its deprecated since CUDA 3.0, are there any substitutes? */
	//CUDA_CALL(cudaGLUnregisterBufferObject(buffer_id_));

	glDeleteBuffers(1, &buffer_id_);
	glDeleteTextures(1, &buffer_id_);
}

void OpenGLSurface::map()
{
	CUDA_CALL(cudaGLMapBufferObject(&pixel_buffer_object_d_, buffer_id_));
}

void OpenGLSurface::unmap()
{
	CUDA_CALL(cudaGLUnmapBufferObject(buffer_id_));
}

std::shared_ptr<OpenGLSurface> OpenGLSurface::create(size_t width, size_t height)
{
	return std::shared_ptr<OpenGLSurface>(new OpenGLSurface(width, height));
}

void OpenGLSurface::draw()
{
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, buffer_id_); 
	GL_CALL(glGetError())
	glBindTexture( GL_TEXTURE_2D, texture_id_);
	GL_CALL(glGetError())
	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, resolution_.x(), resolution_.y(),
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
}