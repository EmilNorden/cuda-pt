#ifndef STATIC_STACK_H_
#define STATIC_STACK_H_

template <typename T, size_t Capacity>
class StaticStack
{
private:
	size_t count_;
	T storage_[Capacity];

	CUDA_CALLABLE void inflate();
 public:
	CUDA_CALLABLE StaticStack();
	CUDA_CALLABLE void push(const T &value);
	CUDA_CALLABLE T pop();
	CUDA_CALLABLE bool empty() const;
};
template <typename T, size_t Capacity>
StaticStack<T, Capacity>::StaticStack()
	: count_(0)
{
}


template <typename T, size_t Capacity>
void StaticStack<T, Capacity>::push(const T &value)
{
	if(count_ < Capacity)
	{
		storage_[count_++] = value;
	}
}

template <typename T, size_t Capacity>
T StaticStack<T, Capacity>::pop()
{
	if(count_ > 0)
	{
		return storage_[--count_];
	}

	return T();
}

template <typename T, size_t Capacity>
bool StaticStack<T, Capacity>::empty() const
{
	return count_ == 0;
}

#endif