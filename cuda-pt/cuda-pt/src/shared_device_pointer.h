#ifndef HUXFLUX_H_
#define HUXFLUX_H_

#include <iostream>
#include <memory>
#include "cuda_runtime.h"

template <typename T>
void FreeDeviceMemory(T* ptr)
{
	std::cout << "FreeDeviceMemory called on address" << std::hex << "0x" << ptr << "\n" << std::dec;
	if(ptr != nullptr)
	{
		cudaFree(ptr);
		ptr = nullptr;
	}
}

template <typename T>
class SharedDevicePointer : std::shared_ptr<T>
{
public:
	SharedDevicePointer();
	SharedDevicePointer(T *ptr);
	T *ptr() const { return std::shared_ptr<T>::get(); }
};

template <typename T>
SharedDevicePointer<T>::SharedDevicePointer()
{
}

template <typename T>
SharedDevicePointer<T>::SharedDevicePointer(T *ptr)
	: std::shared_ptr<T>(ptr, FreeDeviceMemory<T>)
{
	
}

#endif