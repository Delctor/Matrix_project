#pragma once

namespace alge
{
	template<typename T>
	size_t partition(size_t* indices, T* arr, size_t low, size_t high)
	{
		T pivot = arr[indices[low]];
		size_t i = low - 1;
		size_t j = high + 1;

		while (true)
		{
			do
			{
				i++;
			} while (arr[indices[i]] < pivot);

			do
			{
				j--;
			} while (arr[indices[j]] > pivot);

			if (i >= j)
			{
				return j;
			}

			std::swap(indices[i], indices[j]);
		}
	}

	template<typename T>
	void quicksort(size_t* indices, T* arr, size_t low, size_t high)
	{
		if (low < high)
		{
			size_t pivotI = partition(indices, arr, low, high);
			quicksort(indices, arr, low, pivotI);
			quicksort(indices, arr, pivotI + 1, high);
		}
	}
}
