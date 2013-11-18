#include "texture.h"
#include "sdl.h"
#include "error_assertion.h"
//#include "sdl/extensions/SDL_image.h"

Texture::Texture(SDL_Surface *surface)
{
	SDL_assert(surface != nullptr);

	n_colors_ = surface->format->BytesPerPixel;
	width_ = surface->w;
	height_ = surface->h;
	SDL_assert(n_colors_ == 4 || n_colors_ == 3);

	if(n_colors_ == 4)
	{
		if(surface->format->Rmask == 0x000000ff)
			texture_format_ = GL_RGBA;
		else
			texture_format_ = GL_BGRA;
	}
	else if(n_colors_ == 3)
	{
		if(surface->format->Rmask == 0x000000ff)
			texture_format_ = GL_RGB;
		else
			texture_format_ = GL_BGR;
	}

	glGenTextures(1, &texture_);
	glBindTexture(GL_TEXTURE_2D, texture_);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, n_colors_, surface->w, surface->h, 0, texture_format_, GL_UNSIGNED_BYTE, surface->pixels);

}

Texture::~Texture()
{
	glDeleteTextures( 1, &texture_ );
}