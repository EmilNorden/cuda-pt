#ifndef DEVICE_LIST_H_
#define DEVICE_LIST_H_

#include "cuda_runtime.h"
#include "error_assertion.h"

template <typename T>
class DeviceList
{
private:
	T *items_;
	size_t nitems_;
	size_t capacity_;
	CUDA_CALLABLE void expand();
public:
	CUDA_CALLABLE DeviceList();
	CUDA_CALLABLE ~DeviceList();

	CUDA_CALLABLE void add(const T &item);
	CUDA_CALLABLE size_t count() const;
	CUDA_CALLABLE T& operator[] (const int index);
};

template <typename T>
DeviceList<T>::DeviceList()
	: nitems_(0), capacity_(8)
{
	nitems_ = 0;
	items_ = new T[capacity_];
}

template <typename T>
DeviceList<T>::~DeviceList()
{
	if(items_ != nullptr)
	{
		delete[] items_;
		items_ = nullptr;
	}
}

template <typename T>
void DeviceList<T>::expand()
{
	size_t new_capacity = capacity_ * 2;
	T *new_items = new T[new_capacity];

	for(size_t i = 0; i < nitems_; i++)
	{
		new_items[i] = items_[i];
		printf("%d\n", i);
	}
	delete[] items_;
	items_ = new_items;
	capacity_ = new_capacity;
}

template <typename T>
void DeviceList<T>::add(const T &item)
{
	if(nitems_ >= capacity_)
		expand();

	items_[nitems_++] = item;
}

template <typename T>
size_t DeviceList<T>::count() const
{
	return nitems_;
}

template <typename T>
CUDA_CALLABLE T& DeviceList<T>::operator[] (const int index)
{
	return items_[index];
}

#endif