#ifndef SPRITE_H_
#define SPRITE_H_

#include <memory>
#include "vector2.h"
class Texture;

class Sprite
{
private:
	std::shared_ptr<Texture> texture_;
	Vector2d position_;
public:
	Sprite(const std::shared_ptr<Texture> &texture, const Vector2d &position);

	void draw(int viewport_width, int viewport_height) const;

	void set_texture(const std::shared_ptr<Texture> &texture);
	std::shared_ptr<Texture> texture() const { return texture_; }
};

#endif