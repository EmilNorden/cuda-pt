#include "window.h"

#include <SDL.h>


SDLWindow::SDLWindow(int x, int y, size_t width, size_t height, bool fullscreen, const std::string &title)
{
	Uint32 flags = SDL_WINDOW_SHOWN |SDL_WINDOW_OPENGL;

	if(fullscreen)
		flags |= SDL_WINDOW_FULLSCREEN;

	window_ = SDL_CreateWindow(title.c_str(), x, y, width, height, flags);
}

SDLWindow::~SDLWindow()
{
	SDL_DestroyWindow(window_);
}

Vector2i SDLWindow::get_size() const
{
	int width, height;
	SDL_GetWindowSize(window_, &width, &height);

	return Vector2i(width, height);
}