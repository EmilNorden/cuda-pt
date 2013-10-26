#ifndef MOUSE_DEVICE_H_
#define MOUSE_DEVICE_H_

#include <cstdint>

enum MouseButtonState
{
	Pressed = 1,
	Released = 2,
	ChangedThisFrame = 4
};

class MouseDevice
{
private:
	int x_, y_;
	int relative_x_, relative_y_;
	MouseButtonState left_button_;
	MouseButtonState right_button_;
public:
	void frame_update();

	MouseButtonState left_button() const { return left_button_; }
	MouseButtonState right_button() const { return right_button_; }
	int x() const { return x_; }
	int y() const { return y_; }
	int relative_x() const { return relative_x_; }
	int relative_y() const { return relative_y_; }
};

#endif