#ifndef GAMETIME_H_
#define GAMETIME_H_

#include <cstdint>

class GameTime
{
private:
	uint32_t frame_;
	uint32_t real_;
public:
	uint32_t frameTime() const { return frame_; }
	uint32_t realTime() const { return real_; }

	GameTime(uint32_t real)
		: real_(real)
	{
	}

	void frame_update(uint32_t real)
	{
		frame_ = real - real_;
		real_ = real;
	}
};

#endif