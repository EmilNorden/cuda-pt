#include "mouse_device.h"
#include "SDL.h"

void MouseDevice::frame_update()
{
	int x, y;
	
	Uint32 mask = SDL_GetMouseState(&x, &y);

	relative_x_ = x_ - x;
	relative_y_ = y_ - y;

	x_ = x;
	y_ = y;

	MouseButtonState left_, right_;

	if(mask & SDL_BUTTON(1))
		left_ = MouseButtonState::Pressed;
	else
		left_ = MouseButtonState::Released;

	if(mask & SDL_BUTTON(3))
		right_ = MouseButtonState::Pressed;
	else
		right_ = MouseButtonState::Released;

	if(left_ & MouseButtonState::Pressed ^ left_button_ & MouseButtonState::Pressed)
		left_ = static_cast<MouseButtonState>(static_cast<int>(left_) | static_cast<int>(ChangedThisFrame));
	
	if(right_ & MouseButtonState::Pressed ^ right_button_ & MouseButtonState::Pressed)
		right_ = static_cast<MouseButtonState>(static_cast<int>(right_) | static_cast<int>(ChangedThisFrame));

	left_button_ = left_;
	right_button_ = right_;
}