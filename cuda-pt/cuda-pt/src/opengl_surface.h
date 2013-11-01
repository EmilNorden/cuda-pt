#ifndef OPENGL_SURFACE_H_
#define OPENGL_SURFACE_H_

#include <memory>

#include "vector2.h"
#include "vector3.h"

class OpenGLSurface
{
private:
	Vector2i resolution_;
	Vector2i *resolution_d_;
	
	unsigned int buffer_id_;
	unsigned int texture_id_;

	Vector3d *accu_buffer_d_;
	void *pixel_buffer_object_d_;

	OpenGLSurface(size_t width, size_t height);

	// private copy assignment operator and copy constructor
	OpenGLSurface& operator=(OpenGLSurface &other); 
	OpenGLSurface(OpenGLSurface &other);
	
public:
	~OpenGLSurface();
	

	void map();
	void unmap();

	static std::shared_ptr<OpenGLSurface> create(size_t width, size_t height);

	void draw();

	const Vector2i &resolution() const { return resolution_; }
	const Vector2i *resolution_d() const { return resolution_d_; }
	Vector3d *accumulation_buffer_d() const { return accu_buffer_d_; }
	void *pixel_buffer_object_d() const { return pixel_buffer_object_d_; }
};

#endif