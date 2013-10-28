#ifndef FONT_H_
#define FONT_H_

#include <string>
#include "SDL_ttf.h"

class SDLFont
{
private:
	TTF_Font *font_;
	SDLFont(const SDLFont &other);
	SDLFont& operator=(const SDLFont other);
public:
	SDLFont(std::string &path, int size);
	~SDLFont();

	TTF_Font *get() const { return font_; }
};

#endif