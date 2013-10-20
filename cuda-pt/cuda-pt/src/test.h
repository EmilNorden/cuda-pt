#ifndef TEST_H_
#define TEST_H_

#include "cuda_helpers.h"

class foo
{
public:
	CUDA_CALLABLE void bar();
};

#endif