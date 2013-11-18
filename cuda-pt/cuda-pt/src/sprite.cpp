#include "texture.h"
#include "sprite.h"


Sprite::Sprite(const std::shared_ptr<Texture> &texture, const Vector2d &position)
	: texture_(texture), position_(position)
{
}

void Sprite::draw(int viewport_width, int viewport_height) const
{
	glBindTexture(GL_TEXTURE_2D, texture_->gl_texture());
	glLoadIdentity();
	glTranslatef(position_.x(), position_.y(), 0);

	double x = texture_->width() / (double)viewport_width;
	double y = texture_->height() / (double)viewport_height;

	glScalef(x, y, 0);

	glBegin(GL_QUADS);
		glTexCoord2f(0, 0.0f); 
		glVertex3f(0, 1,0);
		glTexCoord2f(0, 1.0f);
		glVertex3f(0, 0,0);
		glTexCoord2f(1.0f, 1.0f);
		glVertex3f(1, 0, 0);
		glTexCoord2f(1.0f, 0.0f);
		glVertex3f(1, 1,0);
	glEnd();
}

void Sprite::set_texture(const std::shared_ptr<Texture> &texture)
{
	texture_ = texture;
}