#ifndef STATIC_UNIFORM_HEAP_H_
#define STATIC_UNIFORM_HEAP_H_

#include "error_assertion.h"

// A simple static (ie will not grow dynamically) heap that can only contain a single type of object.
template <typename T, size_t Capacity>
class StaticUniformHeap
{
private:
	size_t count_;
	size_t first_free_index_;
	T storage_[Capacity];
	bool allocated_[Capacity];
public:
	CUDA_CALLABLE StaticUniformHeap();
	CUDA_CALLABLE T *allocate();
	CUDA_CALLABLE void release(T *item);
};

template <typename T, size_t Capacity>
StaticUniformHeap<T, Capacity>::StaticUniformHeap()
	: count_(0), first_free_index_(0)
{
}

template <typename T, size_t Capacity>
T *StaticUniformHeap<T, Capacity>::allocate()
{
	for(size_t i = first_free_index_; i < Capacity; ++i)
	{
		if(!allocated_[i])
		{
			first_free_index_ = i + 1;
			allocated_[i] = true;
			return &storage_[i];
		}
	}

	return nullptr;
}

template <typename T, size_t Capacity>
void StaticUniformHeap<T, Capacity>::release(T *item)
{
	size_t item_offset = static_cast<size_t>(item - storage_);
	size_t item_index = item_offset / sizeof(T);

	if(item_index < first_free_index_)
		first_free_index_ = item_index;

	allocated_[item_index] = false;
}

#endif