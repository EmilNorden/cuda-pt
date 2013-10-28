#include "font.h"


SDLFont::SDLFont(std::string &path, int size)
{
	font_ = TTF_OpenFont(path.c_str(), size);
}

SDLFont::~SDLFont()
{
	TTF_CloseFont(font_);
}