#ifndef WINDOW_H_
#define WINDOW_H_

#include <string>
#include "vector2.h"

struct SDL_Window;

class SDLWindow
{
private:
	SDL_Window *window_;
public:
	SDLWindow(int x, int y, size_t width, size_t height, bool fullscreen, const std::string &title = "Window");
	~SDLWindow();

	SDL_Window *get() const { return window_; };

	Vector2i get_size() const;
};

#endif