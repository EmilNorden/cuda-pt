#ifndef STACK_H_
#define STACK_H_

#include <cstring>
#include "error_assertion.h"

template <typename T>
class Stack
{
private:
	size_t count_;
	size_t capacity_;
	T *storage_;

	CUDA_CALLABLE void inflate();
 public:
	CUDA_CALLABLE Stack(size_t capacity, T *storage);
	CUDA_CALLABLE explicit Stack(size_t capacity);
	CUDA_CALLABLE ~Stack();

	CUDA_CALLABLE void push(const T &value);
	CUDA_CALLABLE T pop();
	CUDA_CALLABLE bool empty() const;
};

template <typename T>
Stack<T>::Stack(size_t capacity, T *storage)
	: capacity_(capacity), storage_(memory)
{
}

template <typename T>
Stack<T>::Stack(size_t capacity)
	: capacity_(capacity), count_(0)
{
	storage_ = new T[capacity_];
}

template <typename T>
Stack<T>::~Stack()
{
	delete[] storage_;
}

template <typename T>
void Stack<T>::push(const T &value)
{
	if(count_ == capacity_)
		inflate();

	storage_[count_++] = value;
}

template <typename T>
T Stack<T>::pop()
{
	if(empty())
		return T();
	else
		return storage_[--count_];
}

template <typename T>
bool Stack<T>::empty() const
{
	return count_ == 0;
}

template <typename T>
void Stack<T>::inflate()
{
	size_t new_capacity = capacity_ * 2;
	T *inflated = new T[new_capacity];
	memcpy(inflated, storage_, sizeof(T) * capacity_);
	delete[] storage_;

	capacity_ = new_capacity;
	storage_ = inflated;
}

#endif