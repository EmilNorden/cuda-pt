#ifndef WINDOW_H_
#define WINDOW_H_

#include <string>

struct SDL_Window;

class SDLWindow
{
private:
	SDL_Window *window_;
public:
	SDLWindow(int x, int y, size_t width, size_t height, const std::string &title = "Window");
	~SDLWindow();

	SDL_Window *get() const { return window_; };
};

#endif