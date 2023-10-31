#pragma once
#include <vectorDouble.h>
#include <vectorFloat.h>
#include <vectorUint8_t.h>
#include <vectorUint64_t.h>
#include <vectorInt.h>
#include <matrixDouble.h>
#include <matrixFloat.h>
#include <matrixUint8_t.h>

namespace alge
{
	template<typename T>
	inline size_t upper_bound(vector<T>& vector1, size_t left, size_t right, T num)
	{
		T* data1 = vector1._data;
		while (left <= right)
		{
			size_t mid = left + (right - left) / 2;

			if (data1[mid] <= num)
			{
				left = mid + 1;
			}
			else
			{
				right = mid - 1;
			}
		}
		return left;
	}

	template<typename T>
	inline size_t lower_bound(vector<T>& vector1, size_t left, size_t right, T num)
	{
		T* data1 = vector1._data;

		while (left <= right)
		{
			size_t mid = left + (right - left) / 2;

			if (data1[mid] < num)
			{
				left = mid + 1;
			}
			else
			{
				right = mid - 1;
			}
		}
		return left;
	}

}