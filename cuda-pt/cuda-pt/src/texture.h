#ifndef TEXTURE_H_
#define TEXTURE_H_

#include "sdl.h"
#include <GL/glew.h>

struct SDL_Surface;

class Texture
{
private:
	GLuint texture_;
	GLint n_colors_;
	GLenum texture_format_;
	int width_;
	int height_;
public:
	Texture(SDL_Surface *surface);
	~Texture();

	int width() const { return width_; }
	int height() const { return height_; }
	GLuint gl_texture() const { return texture_; }
};

#endif