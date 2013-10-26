#include "window.h"

#include <SDL.h>


SDLWindow::SDLWindow(int x, int y, size_t width, size_t height, const std::string &title)
{
	window_ = SDL_CreateWindow(title.c_str(), x, y, width, height, SDL_WINDOW_SHOWN |SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN);
}

SDLWindow::~SDLWindow()
{
	SDL_DestroyWindow(window_);
}